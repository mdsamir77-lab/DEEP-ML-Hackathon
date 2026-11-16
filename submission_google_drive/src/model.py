"""Model factory and checkpoint loading for timm / torchvision backbones.

Two backends are supported:

``timm``
    Any timm architecture (EfficientNet, Xception, ConvNeXt, ViT, ...).
    Pretrained ImageNet weights are fetched through a configurable mirror
    (``HF_MIRROR`` / default ``https://hf-mirror.com``) because
    ``huggingface.co`` is not always reachable.

``torchvision``
    ``torchvision.models`` architectures (ResNet, EfficientNet, MobileNet,
    ViT, ...). Weights are read from the local torch hub cache first and only
    downloaded (through ``requests``/certifi) when missing, which keeps
    training usable on machines whose Windows certificate store is broken.

Both backends produce a single-logit binary classifier: ``Dropout(p) ->
Linear(in_features, 1)``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import timm
import torch
from torchvision.models import get_model, get_model_weights

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_checkpoint  # noqa: E402

DEFAULT_ARCH = "efficientnet_b0"
DEFAULT_MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
WEIGHT_CACHE = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
BACKENDS = ("auto", "timm", "torchvision")


def _pretrained_cfg(arch: str) -> Dict[str, Any]:
    try:
        return dataclasses.asdict(timm.get_pretrained_cfg(arch))
    except Exception:
        return {}


def _download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    """Stream ``url`` to ``destination`` with a progress bar."""
    import requests
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as handle, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=f"download {destination.name}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    tmp_path.replace(destination)
    return destination


def download_pretrained_weights(arch: str, mirror: Optional[str] = None) -> Optional[Path]:
    """Return a local file with ImageNet weights for ``arch`` (or ``None``)."""
    cfg = _pretrained_cfg(arch)
    repo = cfg.get("hf_hub_id")
    if not repo:
        return None

    candidates = []
    if cfg.get("hf_hub_filename"):
        candidates.append(cfg["hf_hub_filename"])
    candidates.extend(["model.safetensors", "pytorch_model.bin"])

    mirror = mirror or DEFAULT_MIRROR
    for filename in candidates:
        local_path = WEIGHT_CACHE / f"{repo.replace('/', '__')}__{filename}"
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        for endpoint in (mirror, "https://huggingface.co"):
            if not endpoint:
                continue
            url = f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}"
            try:
                return _download_file(url, local_path, timeout=30)
            except Exception:
                continue
    return None


def _read_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_pretrained_weights(model: torch.nn.Module, arch: str, verbose: bool = True) -> bool:
    """Load ImageNet weights into ``model`` (head is skipped on shape mismatch)."""
    weights_path = download_pretrained_weights(arch)
    if weights_path is None:
        if verbose:
            print(f"[model] no pretrained weights found for '{arch}', training from scratch", flush=True)
        return False

    state_dict = _read_state_dict(weights_path)
    state_dict = {key.removeprefix("module.").removeprefix("model."): value for key, value in state_dict.items()}
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = len(state_dict) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    if verbose:
        print(
            f"[model] loaded {len(compatible)} pretrained tensors from {weights_path.name} "
            f"(skipped {skipped}: head / size mismatch)",
            flush=True,
        )
    return True


# ---------------------------------------------------------------------------
# torchvision backend
# ---------------------------------------------------------------------------
def torchvision_weights(arch: str) -> Optional[Any]:
    """Return the default torchvision ``Weights`` enum member for ``arch``."""
    try:
        return get_model_weights(arch).DEFAULT
    except Exception:
        return None


def torchvision_weight_path(arch: str) -> Optional[Path]:
    """Local torch hub cache path for the default torchvision weights of ``arch``."""
    weights = torchvision_weights(arch)
    url = getattr(weights, "url", None)
    if not url:
        return None
    return WEIGHT_CACHE / url.rsplit("/", 1)[-1]


def download_torchvision_weights(arch: str) -> Optional[Path]:
    """Return local ImageNet weights for ``arch``, downloading only if missing.

    ``torch.hub`` relies on the Windows certificate store, which is broken on
    some machines (``ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]``). Streaming the
    file through ``requests`` (certifi CA bundle) avoids that failure mode.
    """
    path = torchvision_weight_path(arch)
    if path is None:
        return None
    if path.exists() and path.stat().st_size > 0:
        return path
    weights = torchvision_weights(arch)
    try:
        return _download_file(weights.url, path, timeout=60)
    except Exception as exc:
        print(f"[model] could not download torchvision weights for '{arch}': {exc}", flush=True)
        return None


def _rebuild_sequential_head(
    head: torch.nn.Sequential, num_classes: int, dropout: float
) -> Tuple[torch.nn.Sequential, int]:
    """Replace the last ``Linear`` of a head and normalise its dropout rate."""
    layers = list(head.children())
    linear_positions = [i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear)]
    if not linear_positions:
        raise ValueError("Sequential classifier head does not contain a Linear layer")
    last = linear_positions[-1]
    in_features = int(layers[last].in_features)

    rebuilt: list = []
    for index, layer in enumerate(layers):
        if index == last:
            rebuilt.append(torch.nn.Dropout(dropout))
            rebuilt.append(torch.nn.Linear(in_features, num_classes))
        elif isinstance(layer, torch.nn.Dropout):
            rebuilt.append(torch.nn.Dropout(dropout))
        else:
            rebuilt.append(layer)
    return torch.nn.Sequential(*rebuilt), in_features


def attach_binary_head(
    model: torch.nn.Module,
    arch: str,
    num_classes: int = 1,
    dropout: float = 0.4,
) -> int:
    """Swap the ImageNet classifier for ``Dropout(dropout) -> Linear(in, num_classes)``.

    Handles the ``fc`` (ResNet), ``classifier`` (EfficientNet / MobileNet) and
    ``heads`` / ``head`` (VisionTransformer / Swin) conventions. Returns the
    number of input features of the new head.
    """
    fc = getattr(model, "fc", None)
    if isinstance(fc, torch.nn.Linear):
        in_features = int(fc.in_features)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(in_features, num_classes)
        )
        return in_features

    for attribute in ("classifier", "heads", "head"):
        head = getattr(model, attribute, None)
        if isinstance(head, torch.nn.Sequential):
            new_head, in_features = _rebuild_sequential_head(head, num_classes, dropout)
            setattr(model, attribute, new_head)
            return in_features
        if isinstance(head, torch.nn.Linear):
            in_features = int(head.in_features)
            setattr(
                model,
                attribute,
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout), torch.nn.Linear(in_features, num_classes)
                ),
            )
            return in_features

    # Some backbones nest the classifier one level deeper (e.g. a wrapper module
    # whose ``classifier``/``head`` attribute is not a Sequential/Linear). Walk the
    # last ``Linear`` in the module tree and swap only that layer, leaving the
    # surrounding flattening/normalisation layers untouched.
    last_linear_name: Optional[str] = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_name = name
    if last_linear_name is None:
        raise ValueError(f"Could not locate a classifier head on torchvision model '{arch}'")

    parent_name, _, child_name = last_linear_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    last_linear = getattr(parent, child_name)
    setattr(
        parent,
        child_name,
        torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(last_linear.in_features), num_classes),
        ),
    )
    return int(last_linear.in_features)


def build_torchvision_model(
    arch: str,
    pretrained: bool = True,
    num_classes: int = 1,
    dropout: float = 0.4,
    verbose: bool = True,
) -> torch.nn.Module:
    """Build a torchvision backbone with a single-logit binary head."""
    try:
        model = get_model(arch, weights=None)
    except Exception as exc:  # unknown architecture for torchvision
        raise ValueError(
            f"'{arch}' is not a torchvision architecture; use --backend timm ({exc})"
        ) from exc

    if pretrained:
        weights_path = download_torchvision_weights(arch)
        if weights_path is not None:
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if verbose:
                print(
                    f"[model] loaded torchvision ImageNet weights from {weights_path.name} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})",
                    flush=True,
                )
        elif verbose:
            print(f"[model] torchvision weights for '{arch}' unavailable; random init", flush=True)

    in_features = attach_binary_head(model, arch, num_classes=num_classes, dropout=dropout)
    if verbose:
        print(
            f"[model] torchvision {arch}: Dropout({dropout}) + Linear({in_features}, {num_classes})",
            flush=True,
        )
    return model


def resolve_backend(arch: str, backend: str = "auto") -> str:
    """Resolve ``auto`` to the backend that can actually build ``arch``.

    A few architecture names are torchvision-only: ``swin_t`` (timm calls it
    ``swin_tiny_patch4_window7_224``) and ``convnext_tiny`` also exist in timm
    but with different weight tensors.  So the rule is: if torchvision knows the
    name, use torchvision; only fall back to timm for names torchvision does not
    expose.  This prevents ``timm.create_model('swin_t')`` from raising
    ``RuntimeError: Unknown model`` when ``--backend auto`` is used.
    """
    if backend in ("timm", "torchvision"):
        return backend
    if torchvision_weights(arch) is not None:
        return "torchvision"
    return "timm"


def build_model(
    arch: str = DEFAULT_ARCH,
    pretrained: bool = True,
    num_classes: int = 1,
    dropout: float = 0.4,
    drop_path_rate: float = 0.0,
    backend: str = "auto",
    verbose: bool = True,
) -> torch.nn.Module:
    """Create a single-logit binary classifier on top of an ImageNet backbone.

    ``backend='torchvision'`` builds the architecture through
    ``torchvision.models`` and loads cached ImageNet weights; ``backend='timm'``
    uses ``timm.create_model`` (which applies ``dropout`` before its own head).
    ``backend='auto'`` prefers torchvision when the matching weights are already
    cached locally and falls back to timm otherwise.
    """
    resolved = resolve_backend(arch, backend)
    if resolved == "torchvision":
        if verbose:
            print(f"[model] backend=torchvision arch={arch}", flush=True)
        return build_torchvision_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            verbose=verbose,
        )

    if verbose:
        print(f"[model] backend=timm arch={arch}", flush=True)
    kwargs: Dict[str, Any] = {"pretrained": False, "num_classes": num_classes}
    if drop_path_rate and drop_path_rate > 0:
        kwargs["drop_path_rate"] = float(drop_path_rate)
    try:
        model = timm.create_model(arch, drop_rate=float(dropout), **kwargs)
    except TypeError:
        # Some architectures (e.g. legacy xception) do not expose drop_rate.
        model = timm.create_model(arch, **kwargs)

    if pretrained:
        loaded = load_pretrained_weights(model, arch, verbose=verbose)
        if not loaded:
            # Last resort: let timm resolve the weights itself (needs Hub access).
            try:
                model = timm.create_model(
                    arch, pretrained=True, num_classes=num_classes, drop_rate=float(dropout)
                )
            except Exception as exc:  # pragma: no cover - offline environments
                print(f"[model] pretrained weights unavailable ({exc}); random init", flush=True)
    return model




def build_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> Tuple[list, list]:
    """Split parameters into decay / no-decay groups (norms and biases excluded)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def load_model_from_checkpoint(
    weights_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild the architecture stored in a checkpoint and load its weights."""
    checkpoint = load_checkpoint(weights_path, map_location="cpu")
    arch = checkpoint.get("arch", DEFAULT_ARCH)
    config = checkpoint.get("config", {}) or {}
    backend = checkpoint.get("backend") or config.get("backend") or "timm"
    model = build_model(
        arch=arch,
        pretrained=False,
        num_classes=1,
        dropout=float(config.get("dropout", 0.0)),
        drop_path_rate=0.0,
        backend=backend,
        verbose=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def describe_head(model: torch.nn.Module) -> str:
    """Human readable description of the classifier head (works for both backends)."""
    for attribute in ("get_classifier", "fc", "classifier", "heads", "head"):
        head = getattr(model, attribute, None)
        if head is None:
            continue
        if callable(head) and not isinstance(head, torch.nn.Module):
            try:
                head = head()
            except Exception:
                continue
        if isinstance(head, torch.nn.Module):
            return f"{attribute}: {head}"
    return "unknown"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check model construction / weight loading")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--backend", default="auto", choices=BACKENDS)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()

    net = build_model(args.arch, pretrained=args.pretrained, backend=args.backend)
    with torch.no_grad():
        out = net(torch.randn(2, 3, 224, 224))
    print("arch:", args.arch, "pretrained:", args.pretrained, "output:", tuple(out.shape))
    print("params:", sum(p.numel() for p in net.parameters()))
    print("head:", describe_head(net))


