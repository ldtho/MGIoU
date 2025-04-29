# mgio3d/mgio.py
from __future__ import annotations

import torch
from torch import Tensor, nn

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

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:  # noqa: D401
        """Compute per-sample MGIoU loss.

        Parameters
        ----------
        pred, target : Tensor
            Corner tensors of identical shape ``[B, 8, 3]``.

        Returns
        -------
        Tensor
            Loss for each sample ``[B]``.
        """
        if pred.shape != target.shape:
            raise ValueError("`pred` and `target` must share shape [B, 8, 3].")
        loss = (1.0 - _torch_vmap(self._mgiou_single)(pred, target)) * 0.5
        return loss

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
        overlaps = []
        for axis in axes:
            min1, max1 = self._project(c1, axis)
            min2, max2 = self._project(c2, axis)
            inter = torch.minimum(max1, max2) - torch.maximum(min1, min2)
            union = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
            overlaps.append(inter / union)
        return torch.mean(torch.stack(overlaps))


# --------------------------------------------------------------------------- #
#                               2-D MGIoU (boxes)                             #
# --------------------------------------------------------------------------- #
class MGIoU2D(nn.Module):
    r"""MGIoU loss for rotated rectangles parameterised as ``(x, y, w, h, θ)``.

    *Input tensors* ``pred`` and ``target`` are shape ``[B, 5]``.
    The loss is reduced according to ``reduction`` and multiplied by
    ``loss_weight`` to match mm-Detection style.
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'.")
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer(
            "_unit_square",
            torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32),
        )

    # --------------------------------------------------------------------- #
    #                                API                                    #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> Tensor:
        """Compute MGIoU loss for a batch of rotated rectangles.

        Parameters
        ----------
        pred, target : Tensor
            *(B, 5)* tensors ``(x, y, w, h, θ)`` (θ in *radians*).
        weight : Tensor, optional
            Element-wise weights broadcastable to ``B``.
        reduction_override : {'none', 'mean', 'sum'}, optional
            Use instead of constructor reduction.
        avg_factor : float, optional
            Divisor applied after reduction (mm-Detection compat).

        Returns
        -------
        Tensor
            Scalar loss (unless ``reduction='none'``).
        """
        red = reduction_override or self.reduction
        if pred.shape != target.shape or pred.shape[-1] != 5:
            raise ValueError("Expected shape (B, 5) for both pred & target.")

        # ------------------------------------------------------------------ #
        # Step 1: mask degenerate GTs → fallback to L1
        device, B = pred.device, pred.size(0)
        all_zero = (target.abs().sum(dim=1) == 0)
        losses = pred.new_zeros(B)

        if all_zero.any():
            l1 = torch.nn.functional.l1_loss(pred[all_zero], target[all_zero], reduction="none")
            losses[all_zero] = l1.sum(dim=1)

        # ------------------------------------------------------------------ #
        # Step 2: MGIoU on valid samples
        mask = ~all_zero
        if mask.any():
            corners_pred = self._rect_to_corners(pred[mask])
            corners_tgt = self._rect_to_corners(target[mask])

            axes = torch.cat(
                (self._rect_axes(corners_pred), self._rect_axes(corners_tgt)), dim=1
            )  # [N, 4, 2]

            proj1 = (corners_pred @ axes.transpose(1, 2))  # [N, 4, 4]
            proj2 = (corners_tgt @ axes.transpose(1, 2))

            min1, max1 = proj1.min(dim=1).values, proj1.max(dim=1).values
            min2, max2 = proj2.min(dim=1).values, proj2.max(dim=1).values

            inter = torch.minimum(max1, max2) - torch.maximum(min1, min2)
            union = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
            overlap = inter / union
            losses[mask] = (1.0 - overlap.mean(dim=-1)) * 0.5

        # ------------------------------------------------------------------ #
        # Step 3: weighting / reduction
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            losses = losses * weight
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
        avg_factor = float(avg_factor or B)
        loss = self._reduce(losses, reduction=red) / avg_factor
        return loss * self.loss_weight

    # ---------------------------- helpers -------------------------------- #
    def _rect_to_corners(self, boxes: Tensor) -> Tensor:
        trans, wh, angle = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
        base = self._unit_square.to(boxes)
        corners = base.unsqueeze(0) * (wh * 0.5).unsqueeze(1)
        cos_a, sin_a = angle.cos(), angle.sin()
        rot = torch.stack(
            (torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)), dim=1
        )  # [B, 2, 2]
        return torch.bmm(corners, rot) + trans.unsqueeze(1)  # [B, 4, 2]

    @staticmethod
    def _rect_axes(corners: Tensor) -> Tensor:
        # edges 0-1 and 0-3  → normals (dy,-dx)
        e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]
        return torch.stack((-e1[..., 1:], e1[..., :1], -e2[..., 1:], e2[..., :1]), dim=1).view(
            -1, 4, 2
        )

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

    def __init__(self, convex_weight: float = 0.0):
        """
        Parameters
        ----------
        convex_weight : float, default 0.0
            Weight applied to the convexity penalty term.
            • 0.0  →  ignore convexity (pure MGIoU)
            • 1.0  →  equal weight with MGIoU
        """
        super().__init__()
        self.convex_weight = convex_weight

    # ------------------------------------------------------------------ #
    #                               API                                   #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        visible_mask: Tensor | None = None,
    ) -> Tensor:
        if pred.shape != target.shape or pred.shape[-2:] != (4, 2):
            raise ValueError("Expect pred & target shape (B, 4, 2).")

        if visible_mask is not None:
            mask = visible_mask.bool().squeeze()
            pred, target = pred[mask], target[mask]

        mgiou = (1.0 - _torch_vmap(self._mgiou_single)(pred, target)) * 0.5

        if self.convex_weight == 0.0:
            return mgiou

        # convexity_loss ∈ [0, 1], lower is better (0 = strictly convex)
        convex_loss_pred = self._convexity_loss(pred)

        return mgiou + self.convex_weight * convex_loss_pred

    # ------------------------------------------------------------------ #
    #                       MGIoU internals                               #
    # ------------------------------------------------------------------ #
    def _mgiou_single(self, c1: Tensor, c2: Tensor) -> Tensor:
        axes = torch.cat((self._candidate_axes(c1), self._candidate_axes(c2)))  # [8,2]
        overlaps = []
        for axis in axes:
            min1, max1 = (c1 @ axis).min(), (c1 @ axis).max()
            min2, max2 = (c2 @ axis).min(), (c2 @ axis).max()
            inter = torch.minimum(max1, max2) - torch.maximum(min1, min2)
            union = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
            overlaps.append(inter / union)
        return torch.mean(torch.stack(overlaps))

    @staticmethod
    def _candidate_axes(corners: Tensor) -> Tensor:
        center = corners.mean(dim=0, keepdim=True)
        angles = torch.atan2(corners[:, 1] - center[0, 1], corners[:, 0] - center[0, 0])
        corners = corners[angles.argsort()]  # clockwise
        edges = torch.vstack((corners[1:] - corners[:-1], corners[:1] - corners[-1:]))
        normals = torch.stack((edges[:, 1], -edges[:, 0]), dim=1)
        return normals  # [4,2]

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
    """Return pairwise MGIoU^{-} for a batch of quadrangles ``[B, 4, 2]``."""

    def forward(self, boxes: Tensor) -> Tensor:  # noqa: D401
        if boxes.ndim != 3 or boxes.shape[-2:] != (4, 2):
            raise ValueError("Input must be (B, 4, 2) quadrangles.")
        pairwise = _torch_vmap(lambda b1: _torch_vmap(self._mgiou_single)(boxes))(boxes)
        diag_idx = torch.arange(boxes.size(0), device=boxes.device)
        pairwise[diag_idx, diag_idx] = 0.0  # self-IoU irrelevant
        return pairwise

    # --------------------------- helpers ---------------------------------- #
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
        overlaps = []
        for axis in axes:
            min1, max1 = self._project(c1, axis)
            min2, max2 = self._project(c2, axis)
            inter = torch.minimum(max1, max2) - torch.maximum(min1, min2)
            union = torch.maximum(max1, max2) - torch.minimum(min1, min2) + _EPS
            overlaps.append(inter / union)
        ov = torch.stack(overlaps)
        return torch.clamp(torch.minimum(ov).max(), min=0.0)
