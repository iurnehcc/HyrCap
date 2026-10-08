"""Class-agnostic hard NMS for temporal segments."""

import numpy as np
import torch

try:
    from . import nms_1d_cpu
except ImportError:
    import nms_1d_cpu


def apply_nms(detections, nms_threshold, minimum_score, top_k):
    """Filter ``[start, end, score, label]`` rows with hard temporal NMS."""
    segments = torch.from_numpy(detections[:, :2]).float()
    scores = torch.from_numpy(detections[:, 2]).float()
    labels = torch.from_numpy(detections[:, 3]).float()

    valid = scores > minimum_score
    segments = segments[valid]
    scores = scores[valid]
    labels = labels[valid]
    indices = nms_1d_cpu.nms(
        segments.contiguous(), scores.contiguous(), iou_threshold=float(nms_threshold)
    )
    if top_k > 0:
        indices = indices[:top_k]
    return np.column_stack(
        (
            segments[indices].numpy(),
            scores[indices].numpy(),
            labels[indices].numpy(),
        )
    )
