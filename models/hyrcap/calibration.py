"""Proposal calibration used by HyrCap training and evaluation."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .modules import CPCC, LQE, PRS


FEATURE_VERSION = "hyrcap-prediction-calibration-v4"
FEATURE_NAMES = [
    "score",
    "logit_score",
    "log_duration",
    "relative_duration",
    "relative_center",
    "auxiliary_support",
    "auxiliary_best_score",
    "auxiliary_top3_iou",
    "auxiliary_score_weighted_support",
    "stronger_neighbor_iou",
    "overlap_density",
    "relative_score_rank",
    "left_boundary_support",
    "right_boundary_support",
    "same_class_auxiliary_support",
    "video_log_duration",
]


def temporal_iou(first, second):
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    intersection = np.maximum(
        0,
        np.minimum(first[:, None, 1], second[None, :, 1])
        - np.maximum(first[:, None, 0], second[None, :, 0]),
    )
    union = (
        (first[:, 1] - first[:, 0])[:, None]
        + (second[:, 1] - second[:, 0])[None, :]
        - intersection
    )
    return intersection / np.maximum(union, 1e-8)


def prepare_prediction(record):
    """Construct checkpoint features from detector predictions without GT."""
    segments = np.asarray(record["primary_segments"], np.float32)
    scores = np.asarray(record["primary_scores"], np.float32)
    labels = np.asarray(record["primary_labels"], np.int64)
    auxiliary = np.asarray(record["auxiliary_segments"], np.float32)
    auxiliary_scores = np.asarray(record["auxiliary_scores"], np.float32)
    duration = float(record["duration"])
    length = np.maximum(segments[:, 1] - segments[:, 0], 1e-5)
    center = segments.mean(1)

    overlap = temporal_iou(segments, auxiliary).astype(np.float32)
    support = CPCC.compute_support(segments, auxiliary)
    best = overlap.argmax(1)
    best_score = auxiliary_scores[best]
    top3 = np.sort(overlap, axis=1)[:, -min(3, overlap.shape[1]) :].mean(1)
    weighted = (overlap * auxiliary_scores[None, :]).max(1)

    same = temporal_iou(segments, segments).astype(np.float32)
    np.fill_diagonal(same, 0)
    stronger = (scores[None, :] > scores[:, None]) & (
        labels[None, :] == labels[:, None]
    )
    stronger_support = np.where(stronger, same, 0).max(1)
    density = (same >= 0.5).sum(1) / max(len(scores), 1)
    rank = np.argsort(np.argsort(-scores)).astype(np.float32) / max(len(scores), 1)

    same_auxiliary = support
    if "auxiliary_labels" in record:
        auxiliary_labels = np.asarray(record["auxiliary_labels"], np.int64)
        same_auxiliary = np.where(
            labels[:, None] == auxiliary_labels[None, :], overlap, 0
        ).max(1)

    left_support = np.exp(-np.abs(segments[:, 0] - auxiliary[best, 0]) / length)
    right_support = np.exp(-np.abs(segments[:, 1] - auxiliary[best, 1]) / length)
    bounded = np.clip(scores, 1e-6, 1 - 1e-6)
    features = np.stack(
        [
            scores,
            np.log(bounded / (1 - bounded)),
            np.log1p(length),
            length / duration,
            center / duration,
            support,
            best_score,
            top3,
            weighted,
            stronger_support,
            density,
            rank,
            left_support,
            right_support,
            same_auxiliary,
            np.full_like(scores, np.log1p(duration)),
        ],
        axis=1,
    ).astype(np.float32)
    table = pd.DataFrame(
        {
            "video_id": record["video_name"],
            "pred_label": labels,
            "t_start": segments[:, 0],
            "t_end": segments[:, 1],
            "detector_score": scores,
            "segment_center": center,
            "segment_len": length,
        }
    )
    edges, edge_features = PRS.build_graph(
        table, edge_iou=0.5, topk=8, higher_only=True
    )
    return {
        "video_name": record["video_name"],
        "duration": duration,
        "segments": torch.from_numpy(segments),
        "scores": torch.from_numpy(scores),
        "labels": torch.from_numpy(labels),
        "features": torch.from_numpy(features),
        "support": torch.from_numpy(support),
        "edges": torch.from_numpy(edges),
        "edge_features": torch.from_numpy(edge_features),
    }


def add_training_targets(record, annotations, threshold=0.5):
    """Build localization-quality and duplicate targets from validation GT."""
    targets = np.asarray([item["segment"] for item in annotations], np.float64)
    if targets.size == 0:
        raise ValueError(f'{record["video_name"]}: validation video has no GT events')
    overlap = temporal_iou(record["segments"], targets)
    quality = overlap.max(1).astype(np.float32)
    matched = overlap.argmax(1)
    keep = np.ones(len(quality), np.float32)
    scores = record["scores"].numpy()
    for target in range(len(targets)):
        indices = np.flatnonzero((matched == target) & (quality >= threshold))
        if len(indices):
            keep[indices] = 0
            best = indices[np.argmax(quality[indices] + 0.05 * scores[indices])]
            keep[best] = 1
    return {
        **record,
        "quality_target": torch.from_numpy(quality),
        "keep_target": torch.from_numpy(keep),
    }


def batch_records(records, device, training=False):
    keys = ["features", "scores", "support"]
    if training:
        keys += ["quality_target", "keep_target"]
    batch = {key: torch.cat([record[key] for record in records]).to(device) for key in keys}
    edges = []
    edge_features = []
    offset = 0
    for record in records:
        edges.append(record["edges"] + offset)
        edge_features.append(record["edge_features"])
        offset += len(record["scores"])
    batch["edges"] = torch.cat(edges, dim=1).to(device)
    batch["edge_features"] = torch.cat(edge_features).to(device)
    return batch


class CalibrationPipeline(nn.Module):
    """Apply the three released proposal-scoring modules."""

    def __init__(
        self, feature_mean, feature_std, hidden=64, graph_hidden=96, dropout=0.1
    ):
        super().__init__()
        self.cpcc = CPCC(learned=True, dropout=dropout)
        self.lqe = LQE(len(FEATURE_NAMES), hidden_dim=hidden, dropout=dropout)
        self.prs = PRS(3, hidden_dim=graph_hidden, dropout=dropout)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float())
        self.register_buffer(
            "feature_std", torch.as_tensor(feature_std).float().clamp_min(0.01)
        )

    @classmethod
    def from_checkpoint_data(cls, checkpoint, device):
        if checkpoint.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Calibration checkpoint requires feature version {FEATURE_VERSION}."
            )
        if checkpoint.get("modules_enabled") != {
            "cpcc": True,
            "lqe": True,
            "prs": True,
        }:
            raise ValueError("Expected CPCC, LQE, and PRS weights.")
        config = checkpoint["calibration_config"]
        state = checkpoint["calibrator"]
        model = cls(
            state["feature_mean"],
            state["feature_std"],
            hidden=config["hidden"],
            dropout=config["dropout"],
        ).to(device)
        model.load_state_dict(state, strict=True)
        return model.eval()

    @torch.inference_mode()
    def calibrate_record(self, record, scoring):
        prepared = prepare_prediction(record)
        output = self(batch_records([prepared], self.feature_mean.device))
        quality = output["quality"].cpu().numpy()
        keep = output["keep"].cpu().numpy()
        gate = output["gate"].cpu().numpy()
        score = self.score(
            prepared,
            quality,
            keep,
            gate,
            scoring["cpcc_beta"],
            scoring["lqe_gamma"],
            scoring["prs_gamma"],
        )
        return {
            "segments": record["primary_segments"],
            "scores": torch.from_numpy(score),
            "labels": prepared["labels"],
        }

    def forward(self, batch):
        features = ((batch["features"] - self.feature_mean) / self.feature_std).clamp(
            -10, 10
        )
        gate_logit = self.cpcc(batch["features"][:, [0, 5, 6, 3]])
        quality_logit = self.lqe(features)
        gate = gate_logit.sigmoid()
        quality = quality_logit.sigmoid()
        nodes = torch.stack([batch["scores"], features[:, 2], quality], dim=1)
        rank, winner, duplicate = self.prs(
            nodes, batch["edges"], batch["edge_features"]
        )
        return {
            "gate_logit": gate_logit,
            "gate": gate,
            "quality_logit": quality_logit,
            "quality": quality,
            "rank_logit": rank,
            "winner_logit": winner,
            "duplicate_logit": duplicate,
            "keep": (-duplicate).sigmoid(),
        }

    @staticmethod
    def loss(output, batch):
        quality = batch["quality_target"]
        keep = batch["keep_target"]
        duplicate = 1 - keep
        balance = (1 - duplicate.mean()).clamp_min(0.01) / duplicate.mean().clamp_min(0.001)
        weights = torch.where(
            duplicate > 0.5,
            balance.clamp(max=20),
            torch.ones_like(keep),
        )
        lqe_loss = F.binary_cross_entropy_with_logits(output["quality_logit"], quality)
        cpcc_loss = F.binary_cross_entropy_with_logits(output["gate_logit"], quality)
        prs_loss = (
            F.binary_cross_entropy_with_logits(
                output["duplicate_logit"], duplicate, reduction="none"
            )
            * weights
        ).mean()
        winner_loss = (
            F.binary_cross_entropy_with_logits(
                output["winner_logit"], keep, reduction="none"
            )
            * weights
        ).mean()
        rank_loss = F.binary_cross_entropy_with_logits(output["rank_logit"], quality)
        return (
            lqe_loss
            + 0.25 * cpcc_loss
            + 0.25 * prs_loss
            + 0.1 * winner_loss
            + 0.1 * rank_loss
        )

    def score(self, record, quality, keep, gate, beta, lqe_gamma, prs_gamma):
        if min(beta, lqe_gamma, prs_gamma) <= 0:
            raise ValueError("Calibration coefficients must be positive.")
        frame = pd.DataFrame(
            {
                "score": record["scores"].numpy(),
                "best_iou_ref": record["support"].numpy(),
            }
        )
        consensus = self.cpcc(frame, ratio=1.0, beta=beta, gate=gate)["score"].to_numpy(
            np.float64
        )
        return (
            consensus
            * np.maximum(quality, 1e-6) ** lqe_gamma
            * np.maximum(keep, 1e-6) ** prs_gamma
        ).astype(np.float32)


def fit_calibration(records, database, config, device):
    """Train CPCC, LQE, and PRS from frozen-detector validation predictions."""
    if not records or any(
        database[record["video_name"]]["subset"] != "val" for record in records
    ):
        raise ValueError("Calibration training requires THUMOS14 val videos only.")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    prepared = [
        add_training_targets(
            prepare_prediction(record),
            database[record["video_name"]]["annotations"],
            threshold=config["threshold"],
        )
        for record in records
    ]
    features = torch.cat([record["features"] for record in prepared])
    model = CalibrationPipeline(
        features.mean(0),
        features.std(0),
        hidden=config["hidden"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        indices = np.random.permutation(len(prepared))
        losses = []
        for start in range(0, len(indices), config["batch_size"]):
            chosen = [
                prepared[index]
                for index in indices[start : start + config["batch_size"]]
            ]
            batch = batch_records(chosen, device, training=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite calibration loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config["epochs"]:
            print(f"Calibration {row}", flush=True)
    return model.eval(), history
