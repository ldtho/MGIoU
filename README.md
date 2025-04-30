

# Marginalized Generalized IoU (MGIoU): Unifying Parametric Shape Optimization
A PyTorch implementation of **MGIoU**, **MGIoU⁺**, and **MGIoU⁻** for 2D and 3D object detection and collision-aware trajectory prediction.
<p align="center">
  <a href="https://ldtho.github.io/MGIoU/" target="_blank">
    <img src="https://img.shields.io/badge/Project&nbsp;Page-MGIoU-blue?logo=githubpages&logoColor=white" alt="Project Page">
  </a>
&nbsp;
  <a href="https://arxiv.org/abs/2504.16443" target="_blank">
    <img src="https://img.shields.io/badge/ArXiv-2504.16443-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>

## TODO / Road-map
- [x] **v0.1.0** • Release MGIoU, MGIoU⁺, MGIoU⁻  
- [x] Release PyPI package and samples  
- [ ] Sub-repos for experiments in the paper  
  - [ ] 3D Rotated Object Detection  
  - [ ] 2D Rotated Object Detection  
  - [ ] 2D Quadrangle Detection  

*PRs and feature requests are welcome!*

---

## Installation

### Requirements
- Pytorch (preferred >= 2.0)

### 💡 Option 1 — via **pip** (recommended)

```bash
pip install mgiou          # pulls the latest release from PyPI
```
### 💡 Option 2 — from source
```bash
git clone https://github.com/ldtho/MGIoU.git
cd MGIoU
pip install -e .           # editable install for development
```

## Quickstart

```python
import torch
from mgiou import MGIoU3D, MGIoU2D, MGIoU2DPlus, MGIoU2DMinus

# -------------------------------------------------------------------------
# 3-D MGIoU
# -------------------------------------------------------------------------
B = 4
pred_3d   = torch.rand(B, 8, 3)  # 8 corners per box (v0–v7 in the order described below)
target_3d = torch.rand(B, 8, 3)

# default: mean reduction, fast_mode=False
loss3d = MGIoU3D()  
print("MGIoU-3D loss (mean over batch):", loss3d(pred_3d, target_3d))

# you can change reduction or enable fast projections:
loss3d_sum = MGIoU3D(reduction="sum", fast_mode=True)
print("MGIoU-3D loss (sum):", loss3d_sum(pred_3d, target_3d))

# -------------------------------------------------------------------------
# 2-D rotated rectangles
# -------------------------------------------------------------------------
# one box in (x, y, w, h, θ) format
pred_boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3]], dtype=torch.float32)
tgt_boxes  = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=torch.float32)

# default: "rect" representation (x, y, w, h, θ), mean reduction
loss2d = MGIoU2D()
print("MGIoU-2D loss (rect):", loss2d(pred_boxes, tgt_boxes))

# corner mode: supply (B,4,2) directly
corners1 = loss2d._rect_to_corners(pred_boxes)  # reuse helper
corners2 = loss2d._rect_to_corners(tgt_boxes)
loss2d_corner = MGIoU2D(representation="corner", reduction="none")
print("Per-sample MGIoU-2D (corners, none):", loss2d_corner(corners1, corners2))

# override reduction at call-time
print("Same, but summed over batch:",
      MGIoU2D(reduction="none")(pred_boxes, tgt_boxes, reduction_override="sum"))

# -------------------------------------------------------------------------
# 2-D quadrangles with convexity penalty (MGIoU⁺)
# -------------------------------------------------------------------------
pred_quads   = torch.rand(B, 4, 2)
target_quads = torch.rand(B, 4, 2)

# w=0.3 convexity weight, default mean reduction
loss_plus = MGIoU2DPlus(convex_weight=0.3)
print("MGIoU-quad⁺ loss (mean):", loss_plus(pred_quads, target_quads).item())

# sum reduction, fast mode
loss_plus_sum = MGIoU2DPlus(convex_weight=0.3, fast_mode=True, reduction="sum")
print("MGIoU-quad⁺ loss (sum):", loss_plus_sum(pred_quads, target_quads))

# -------------------------------------------------------------------------
# Pairwise MGIoU⁻ (MGIoU2DMinus)
# -------------------------------------------------------------------------
# Using rotated-rect inputs
pairwise_rect = MGIoU2DMinus(representation="rect")(pred_boxes.repeat(B,1))
print("Pairwise MGIoU⁻ matrix (rect):\n", pairwise_rect)

# Using explicit corners and summing all off-diagonals
pairwise_corner_sum = MGIoU2DMinus(
    representation="corner", reduction="sum"
)(corners1.repeat(B,1,1))
print("Sum of all off-diagonal MGIoU⁻:", pairwise_corner_sum.item())
```


## API at a glance
```text
# 3-D boxes as 8 corner points → per-sample loss
MGIoU3D(
    reduction: str = "mean",     # "none" | "mean" | "sum"
    fast_mode: bool = False
)

# 2-D rotated rectangles or explicit corners → per-sample loss
MGIoU2D(
    representation: str = "rect", # "rect" | "corner"
    reduction: str = "mean",      # "none" | "mean" | "sum"
    loss_weight: float = 1.0,
    fast_mode: bool = False
)

# 2-D arbitrary convex quads + convexity penalty → per-sample loss
MGIoU2DPlus(
    convex_weight: float = 0.0,   # 0 = pure MGIoU, >0 adds convexity penalty
    fast_mode: bool = False,
    reduction: str = "mean"       # "none" | "mean" | "sum"
)

# Pairwise MGIoU⁻ matrix → scalar or tensor
MGIoU2DMinus(
    representation: str = "rect", # "rect" | "corner"
    fast_mode: bool = False,
    reduction: str = "mean"       # "none" | "mean" | "sum"
)
```

## Citation 
If you find this repo useful, please cite
```bibtex
@article{le2025marginalized,
  title={Marginalized Generalized IoU (MGIoU): A Unified Objective Function for Optimizing Any Convex Parametric Shapes},
  author={Le, Duy-Tho and Pham, Trung and Cai, Jianfei and Rezatofighi, Hamid},
  journal={arXiv preprint arXiv:2504.16443},
  year={2025}
}
```

## License
This project is licensed under the terms of the MIT license. 