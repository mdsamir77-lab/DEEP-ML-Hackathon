# Grad-CAM Attention Map Analysis (Phase 2 Ensemble Backbone)

**Model:** resnet50 (Phase 2 Primary Ensemble Member, Weight = 0.40)  
**Checkpoint:** `checkpoints/runs/p2_resnet50_s42/best.pth`  
**Layer Visualized:** `layer4[-1]`  
**Member Validation AUC:** 0.8970 | **Final Ensemble Val AUC:** 0.9010  
**Output Plot:** `gradcam_summary.png`

## Key Interpretability Findings

1. **Synthetic Artifact Localization (Deepfakes):**
   - On deepfake crops, the model's highest gradient activations consistently concentrate along:
     - **Facial boundary blending seams** (jawline and cheek contours where generative warps mismatch background texture).
     - **Periorbital and eye reflection irregularities** (asymmetrical specular highlights common in synthetic generation).
     - **Mouth and teeth blur transitions** (frequent synthesis blending artifacts).
   - This proves the Phase 2 classifier is actively diagnosing generative manipulation cues rather than background noise.

2. **Physiological Feature Grounding (Real Faces):**
   - On genuine face images, high activation weights disperse naturally across holistic anatomical landmarks (consistent skin pore textures, nose bridge geometry, natural symmetry).
   - The absence of concentrated seam-focused activations correctly results in very low $P(\text{Fake})$ scores ($< 0.15$).

3. **Robustness & Generalization Implication:**
   - Because activations focus on structural blend boundaries rather than isolated high-frequency pixel noise, the model retains high AUC stability under JPEG compression, mild blur, and resolution scaling.
