

# MGIoU : Marginalised Generalised IoU losses in 2-D & 3-D
A lightweight PyTorch implementation of **MGIoU** losses for rotated rectangles, arbitrary quadrangles, and 3-D bounding boxes.  
Perfect for camera-LiDAR fusion, aerial imagery, or any setting where classic IoU comes up short.


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
- [x] **v0.1.0** &nbsp;• Code for MGIoU, MGIoU+, and MGIoU- 
- [x] Release PyPI package and sample usage
- [ ] Sub-repositories for experiments shown in the paper
  - [ ] 3D Rotated Object Detection 
  - [ ] 2D Rotated Object Detection
  - [ ] 2D Quadrangle Detection

[//]: # (- [ ] ~~Add support for polygons with *N* > 4~~ *&#40;won’t fix in this repo&#41;*  )

*PRs and feature requests are welcome!*

---

## Installation

### 💡 Option 1 — via **pip** (recommended)

```bash
pip install mgio3d         # pulls the latest release from PyPI
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
from mgiou3d import MGIoU3D, MGIoU2D, MGIoU2DPlus, MGIoU2DMinus

# 3-D MGIoU -----------------------------------------------------------------
B = 4
pred_3d   = torch.rand(B, 8, 3)  # v0, v1, v2, v3, v4, v5, v6, v7
target_3d = torch.rand(B, 8, 3)
#       v4_____________________v5
#        /|                    /|
#       / |                   / |
#      /  |                  /  |
#     /___|_________________/   |
# v0 |    |              v1 |   |
#    |    |                 |   |
#    |    |                 |   |
#    |    |                 |   |
#    |    |_________________|___|
#    |   / v7               |   /v6
#    |  /                   |  /
#    | /                    | /
#    |/_____________________|/
#   v3                     v2
# where **v0-v1**, **v0-v3**, **v0-v4** give the three face normals.
# Please ensure that the points/vertices v0, v1, v3, v4 are in correct order

loss3d = MGIoU3D()
print("MGIoU-3D loss:", loss3d(pred_3d, target_3d).mean())

# 2-D rotated rectangles ----------------------------------------------------
pred_boxes   = torch.tensor([[0, 0, 4, 2, 0.3]], dtype=torch.float32)  # (x,y,w,h,rot)
target_boxes = torch.tensor([[0, 0, 4, 2, 0.0]], dtype=torch.float32)

loss2d = MGIoU2D()
print("MGIoU-2D loss:", loss2d(pred_boxes, target_boxes))

# 2-D quadrangles with convexity penalty ------------------------------------
pred_quads   = torch.rand(B, 4, 2)
target_quads = torch.rand(B, 4, 2)

loss_quad = MGIoU2DPlus(convex_weight=0.3)
print("MGIoU-quad loss:", loss_quad(pred_quads, target_quads).mean())

# Pairwise MGIoU- (Minus) ---------------------------------------------------
pairwise = MGIoU2DMinus()(pred_quads)     # shape (B, B)
print("Pairwise negative MGIoU\n", pairwise)
```


## API at a glance
```text
MGIoU3D()            # → loss per sample for boxes given as 8 corners
MGIoU2D()            # → loss for rotated rectangles (x,y,w,h,θ)
MGIoU2DPlus(w)       # → quadrangle loss + w·convexity_penalty
MGIoU2DMinus()       # → pairwise (B×B) “MGIoU⁻” matrix

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