# mgio3d/mgio.py
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:                       # PyTorch ≥ 2.0
    _torch_vmap = torch.vmap
except AttributeError:     # PyTorch ≤ 1.x  →  requires functorch
    from functorch import vmap as _torch_vmap  # type: ignore
    _torch_vmap = _torch_vmap  # silence mypy

__all__ = [
    "MGIoU3D",
    "MGIoU2D",
    "MGIoU2DPlus",
    "MGIoU2DMinus",
]

_EPS = 0 # numerical stability


# --------------------------------------------------------------------------- #
#                               3-D MGIoU                                     #
# --------------------------------------------------------------------------- #
class MGIoU3D(nn.Module):
    r"""Marginalised Generalised IoU for 3-D axis-aligned or rotated boxes.

    Each box is supplied as its **8 corner points** in XYZ order—
    shape ``[B, 8, 3]`` (batch, corner, xyz).  Corner order must satisfy::

          v4_____________________v5
           /|                    /|
          / |                   / |
         /  |                  /  |
        /___|_________________/   |
    v0 |    |              v1 |   |
       |    |                 |   |
       |    |                 |   |
       |    |                 |   |
       |    |_________________|___|
       |   / v7               |   /v6
       |  /                   |  /
       | /                    | /
       |/_____________________|/
      v3                     v2

    where **v0-v1**, **v0-v3**, **v0-v4** give the three face normals.

    Notes
    -----
    * Uses Separating Axis Theorem (SAT) with face normals only.
    * Loss is scaled to :math:`[0,1]`, where **0 = perfect overlap**.
    """
    def __init__(self, reduction: str = "mean", fast_mode: bool = False):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'.")
        self.fast_mode = fast_mode
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Compute MGIoU loss per sample, then reduce."""
        red = reduction_override or self.reduction
        if pred.shape != target.shape:
            raise ValueError("`pred` and `target` must have shape [B,8,3].")
        # per‐sample loss [B]
        losses = (1.0 - _torch_vmap(self._mgiou_single)(pred, target)) * 0.5
        return self._reduce(losses, red)

    @staticmethod
    def _reduce(loss: Tensor, reduction: str) -> Tensor:
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        # unreachable
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    # ---- internal helpers -------------------------------------------------
    @staticmethod
    def _candidate_axes(corners: Tensor) -> Tensor:
        # three edge vectors emanating from v0
        edge_x = corners[1] - corners[0]
        edge_y = corners[3] - corners[0]
        edge_z = corners[4] - corners[0]
        return torch.stack((edge_x, edge_y, edge_z))  # [3, 3]

    @staticmethod
    def _project(corners: Tensor, axis: Tensor) -> tuple[Tensor, Tensor]:
        scalars = corners @ axis
        return scalars.min(), scalars.max()

    def _mgiou_single(self, c1: Tensor, c2: Tensor) -> Tensor:
        axes = torch.cat((self._candidate_axes(c1), self._candidate_axes(c2)))  # [6, 3]
        mgiou1d = []
        for axis in axes:
            min1, max1 = self._project(c1, axis)
            min2, max2 = self._project(c2, axis)
            if self.fast_mode:
                numerator = torch.minimum(max1, max2) - torch.maximum(min1, min2)
                denominator = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
                mgiou1d.append(numerator / denominator)
            else:
                # Get intersection, union, and convex hull, then compute MGIoU
                inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
                union = (max1 - min1) + (max2 - min2) - inter
                hull = (torch.maximum(max1, max2) - torch.minimum(min1, min2))
                mgiou1d.append(inter / union - (hull - union) / hull)
        return torch.mean(torch.stack(mgiou1d))


# --------------------------------------------------------------------------- #
#                               2-D MGIoU (boxes)                             #
# --------------------------------------------------------------------------- #
class MGIoU2D(nn.Module):
    """MGIoU loss for either rotated rectangles (x,y,w,h,θ) or explicit 4-corner boxes."""
    def __init__(
        self,
        representation: str = "rect",        # "rect" or "corner"
        reduction: str = "mean",
        loss_weight: float = 1.0,
        fast_mode: bool = False,
    ):
        super().__init__()
        if representation not in {"rect", "corner"}:
            raise ValueError("representation must be 'rect' or 'corner'")
        self.representation = representation
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.fast_mode = fast_mode

        # only needed if converting from (x,y,w,h,θ)
        self.register_buffer(
            "_unit_square",
            torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32),
        )

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> Tensor:
        red = reduction_override or self.reduction

        # --- convert or validate inputs ---
        if self.representation == "rect":
            # (B,5) → corners
            if pred.shape != target.shape or pred.shape[-1] != 5:
                raise ValueError("Expected (B,5) boxes for 'rect' mode")
            B = pred.size(0)

            # detect degenerate GTs → fallback to L1 as before
            all_zero = (target.abs().sum(dim=1) == 0)
            losses = pred.new_zeros(B)
            if all_zero.any():
                l1 = F.l1_loss(pred[all_zero], target[all_zero], reduction="none")
                losses[all_zero] = l1.sum(dim=1)

            mask = ~all_zero
            if mask.any():
                # convert to corners
                c1 = self._rect_to_corners(pred[mask])
                c2 = self._rect_to_corners(target[mask])
                losses[mask] = self._mgiou_boxes(c1, c2)

        else:  # "corner" mode
            # expect (B,4,2)
            if pred.ndim != 3 or pred.shape[-2:] != (4, 2):
                raise ValueError("Expected (B,4,2) corners for 'corner' mode")
            if pred.shape != target.shape:
                raise ValueError("pred and target must match shape")
            B = pred.size(0)
            # compute MGIoU on all
            losses = self._mgiou_boxes(pred, target)

        # --- weighting & reduction (common) ---
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            losses = losses * weight
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
        avg_factor = float(avg_factor or B)
        loss = self._reduce(losses, red) / avg_factor
        return loss * self.loss_weight

    def _mgiou_boxes(self, c1: Tensor, c2: Tensor) -> Tensor:
        # c1, c2: (N,4,2)
        axes = torch.cat((self._rect_axes(c1), self._rect_axes(c2)), dim=1)  # [N,4,2]
        proj1 = c1 @ axes.transpose(1, 2)  # [N,4,4]
        proj2 = c2 @ axes.transpose(1, 2)

        mn1, mx1 = proj1.min(dim=1).values, proj1.max(dim=1).values
        mn2, mx2 = proj2.min(dim=1).values, proj2.max(dim=1).values

        if self.fast_mode:
            num = torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)
            den = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2) + _EPS
            giou1d = num / den
        else:
            inter = (torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)).clamp(min=0.0)
            union = (mx1 - mn1) + (mx2 - mn2) - inter
            hull  = (torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2))
            giou1d = inter / union - (hull - union) / hull

        return ((1.0 - giou1d.mean(dim=-1)) * 0.5)

    def _rect_to_corners(self, boxes: Tensor) -> Tensor:
        trans, wh, angle = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
        base = self._unit_square.unsqueeze(0) * (wh * 0.5).unsqueeze(1)  # [B,4,2]
        cos_a, sin_a = angle.cos(), angle.sin()
        rot = torch.stack(
            (torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)),
            dim=1,
        )  # [B,2,2]
        return torch.bmm(base, rot) + trans.unsqueeze(1)  # [B,4,2]

    @staticmethod
    def _rect_axes(corners: Tensor) -> Tensor:
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 3] - corners[:, 0]
        normals = torch.stack((-e1[..., 1:], e1[..., :1], -e2[..., 1:], e2[..., :1]), dim=1)
        return normals.view(-1, 2, 2)

    @staticmethod
    def _reduce(loss: Tensor, reduction: str) -> Tensor:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction!r}")


# --------------------------------------------------------------------------- #
#                     2-D MGIoU for arbitrary quadrangles                     #
# --------------------------------------------------------------------------- #
class MGIoU2DPlus(nn.Module):
    """MGIoU for arbitrary convex quadrangles with an optional convexity loss."""

    def __init__(
        self,
        convex_weight: float = 0.0,
        fast_mode: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'.")
        self.convex_weight = convex_weight
        self.fast_mode = fast_mode
        self.reduction = reduction

    # ------------------------------------------------------------------ #
    #                               API                                   #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred: Tensor,                  # [B,4,2]
        target: Tensor,                # [B,4,2]
        visible_mask: Tensor | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Compute per‐sample MGIoU + convexity, then reduce."""
        if visible_mask is not None:
            mask = visible_mask.bool().squeeze()
            pred, target = pred[mask], target[mask]

        # per‐sample pure MGIoU
        losses = (1.0 - _torch_vmap(self._mgiou_single)(pred, target)) * 0.5

        # add convex‐penalty if requested
        if self.convex_weight != 0.0:
            convex_loss = self._convexity_loss(pred)  # [B]
            losses = losses + self.convex_weight * convex_loss

        # reduce
        red = reduction_override or self.reduction
        return self._reduce(losses, red)

    @staticmethod
    def _reduce(loss: Tensor, reduction: str) -> Tensor:
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    # ------------------------------------------------------------------ #
    #                       MGIoU internals                              #
    # ------------------------------------------------------------------ #
    def _mgiou_single(self, c1: Tensor, c2: Tensor) -> Tensor:
        axes = torch.cat((self._candidate_axes(c1), self._candidate_axes(c2)))
        giou1d = []
        for axis in axes:
            min1, max1 = (c1 @ axis).min(), (c1 @ axis).max()
            min2, max2 = (c2 @ axis).min(), (c2 @ axis).max()
            if self.fast_mode:
                numerator = torch.minimum(max1, max2) - torch.maximum(min1, min2)
                denominator = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
                giou1d.append(numerator / denominator)
            else:
                # Get intersection, union, and convex hull, then compute MGIoU
                inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
                union = (max1 - min1) + (max2 - min2) - inter
                hull = (torch.maximum(max1, max2) - torch.minimum(min1, min2))
                giou1d.append(inter / union - (hull - union) / hull)
        return torch.mean(torch.stack(giou1d))

    @staticmethod
    def _candidate_axes(corners: Tensor) -> Tensor:
        center = corners.mean(dim=0, keepdim=True)
        angles = torch.atan2(corners[:, 1] - center[0, 1], corners[:, 0] - center[0, 0])
        corners = corners[angles.argsort()]  # clockwise
        edges = torch.vstack((corners[1:] - corners[:-1], corners[:1] - corners[-1:]))
        normals = torch.stack((edges[:, 1], -edges[:, 0]), dim=1)
        return normals

    # ------------------------------------------------------------------ #
    #                    Convexity consistency penalty                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convexity_loss(polygons: Tensor) -> Tensor:
        """Mean ε where ε>0 indicates non-convex vertices (0 = perfectly convex)."""
        B, N, _ = polygons.shape  # N=4 for quadrangles but works for any N≥3
        v_prev = polygons[:, torch.arange(N) - 1]          # i-1
        v_curr = polygons
        v_next = polygons[:, (torch.arange(N) + 1) % N]    # i+1

        edge1 = v_prev - v_curr
        edge2 = v_next - v_curr
        cross = edge1[..., 0] * edge2[..., 1] - edge1[..., 1] * edge2[..., 0]  # (B,N)

        sign_ref = torch.where(cross[:, 0:1].abs() <= _EPS, torch.ones_like(cross[:, 0:1]), cross[:, 0:1]).sign()
        penalty = torch.clamp(-sign_ref * cross, min=0.0)  # negative => non-convex
        return penalty.mean(dim=1)  # [B]



# --------------------------------------------------------------------------- #
#                           MGIoU "Minus" (pairwise)                          #
# --------------------------------------------------------------------------- #
class MGIoU2DMinus(nn.Module):
    """Pairwise MGIoU⁻ for either (x,y,w,h,θ) or (4,2) corners, with reduction."""
    def __init__(
        self,
        representation: str = "rect",       # "rect" or "corner"
        fast_mode: bool = False,
        reduction: str = "mean",            # default mean over off‐diagonals
    ):
        super().__init__()
        if representation not in {"rect", "corner"}:
            raise ValueError("representation must be 'rect' or 'corner'")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'.")
        self.representation = representation
        self.fast_mode = fast_mode
        self.reduction = reduction

        # needed only if we convert from (x,y,w,h,θ)
        self.register_buffer(
            "_unit_square",
            torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32),
        )

    def forward(
        self,
        boxes: Tensor,                       # [B,5] or [B,4,2]
        reduction_override: str | None = None,
    ) -> Tensor:
        red = reduction_override or self.reduction

        # 1) get corner‐representation
        if self.representation == "rect":
            if boxes.ndim != 2 or boxes.shape[-1] != 5:
                raise ValueError("Expected (B,5) for 'rect' mode.")
            corners = self._rect_to_corners(boxes)
        else:
            if boxes.ndim != 3 or boxes.shape[-2:] != (4, 2):
                raise ValueError("Expected (B,4,2) for 'corner' mode.")
            corners = boxes

        # 2) pairwise via nested vmap → [B,B]
        pairwise = _torch_vmap(
            lambda c1: _torch_vmap(lambda c2: self._mgiou_single(c1, c2))(corners)
        )(corners)

        # zero out diag
        B = pairwise.size(0)
        idx = torch.arange(B, device=pairwise.device)
        pairwise[idx, idx] = 0.0

        # 3) reduction
        if red == "none":
            return pairwise
        # mask out diagonal
        mask = torch.ones_like(pairwise, dtype=torch.bool)
        mask[idx, idx] = False
        vals = pairwise[mask]
        return vals.mean() if red == "mean" else vals.sum()

    def _rect_to_corners(self, boxes: Tensor) -> Tensor:
        # same as in MGIoU2D above
        trans, wh, angle = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
        base = self._unit_square.unsqueeze(0) * (wh * 0.5).unsqueeze(1)
        cos_a, sin_a = angle.cos(), angle.sin()
        rot = torch.stack(
            (torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)),
            dim=1,
        )
        return torch.bmm(base, rot) + trans.unsqueeze(1)

    @staticmethod
    def _edge_axes(c: Tensor) -> Tensor:
        e1, e2 = c[1] - c[0], c[3] - c[0]
        return torch.stack((e1, e2))  # [2,2]

    @staticmethod
    def _project(corners: Tensor, axis: Tensor) -> tuple[Tensor, Tensor]:
        scalars = corners @ axis
        return scalars.min(), scalars.max()

    def _mgiou_single(self, c1: Tensor, c2: Tensor) -> Tensor:
        axes = torch.cat((self._edge_axes(c1), self._edge_axes(c2)))  # [4,2]
        giou1d = []
        for axis in axes:
            min1, max1 = self._project(c1, axis)
            min2, max2 = self._project(c2, axis)
            if self.fast_mode:
                numerator = torch.minimum(max1, max2) - torch.maximum(min1, min2)
                denominator = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
                giou1d.append(numerator / denominator)
            else:
                # Get intersection, union, and convex hull, then compute MGIoU
                inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
                union = (max1 - min1) + (max2 - min2) - inter
                hull = (torch.maximum(max1, max2) - torch.minimum(min1, min2))
                giou1d.append(inter / union - (hull - union) / hull)
        giou1d = torch.stack(giou1d)
        return giou1d.min().clamp(min=0.0)
