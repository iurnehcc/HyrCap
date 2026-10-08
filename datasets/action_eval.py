"""Class-agnostic temporal localization evaluation for THUMOS14."""

import numpy as np
import pandas as pd

from util.nms import apply_nms


COLUMNS = ["video-id", "t-start", "t-end", "score", "cls"]


def _segment_iou(target, candidates):
    intersection = (
        np.minimum(target[1], candidates[:, 1])
        - np.maximum(target[0], candidates[:, 0])
    ).clip(0)
    union = candidates[:, 1] - candidates[:, 0] + target[1] - target[0] - intersection
    return intersection / union


def _interpolated_ap(precision, recall):
    precision = np.hstack(([0], precision, [0]))
    recall = np.hstack(([0], recall, [1]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    points = np.where(recall[1:] != recall[:-1])[0] + 1
    return np.sum((recall[points] - recall[points - 1]) * precision[points])


def average_precision(ground_truth, predictions, thresholds):
    """Compute one-to-one temporal detection AP at every tIoU threshold."""
    ap = np.zeros(len(thresholds))
    if predictions.empty:
        return ap

    predictions = predictions.iloc[
        predictions["score"].to_numpy().argsort()[::-1]
    ].reset_index(drop=True)
    grouped_ground_truth = ground_truth.groupby("video-id")
    locks = np.full((len(thresholds), len(ground_truth)), -1.0)
    true_positive = np.zeros((len(thresholds), len(predictions)))
    false_positive = np.zeros_like(true_positive)

    for prediction_index, prediction in predictions.iterrows():
        try:
            video_ground_truth = grouped_ground_truth.get_group(
                prediction["video-id"]
            ).reset_index()
        except KeyError:
            false_positive[:, prediction_index] = 1
            continue

        overlaps = _segment_iou(
            prediction[["t-start", "t-end"]].to_numpy(),
            video_ground_truth[["t-start", "t-end"]].to_numpy(),
        )
        ordered = overlaps.argsort()[::-1]
        for threshold_index, threshold in enumerate(thresholds):
            for ground_truth_index in ordered:
                if overlaps[ground_truth_index] < threshold:
                    false_positive[threshold_index, prediction_index] = 1
                    break
                lock_index = int(video_ground_truth.loc[ground_truth_index, "index"])
                if locks[threshold_index, lock_index] >= 0:
                    continue
                true_positive[threshold_index, prediction_index] = 1
                locks[threshold_index, lock_index] = prediction_index
                break
            if (
                false_positive[threshold_index, prediction_index] == 0
                and true_positive[threshold_index, prediction_index] == 0
            ):
                false_positive[threshold_index, prediction_index] = 1

    true_positive = np.cumsum(true_positive, axis=1)
    false_positive = np.cumsum(false_positive, axis=1)
    recall = true_positive / float(len(ground_truth))
    precision = true_positive / (true_positive + false_positive)
    for threshold_index in range(len(thresholds)):
        ap[threshold_index] = _interpolated_ap(
            precision[threshold_index], recall[threshold_index]
        )
    return ap


class TemporalDetectionEvaluator:
    """Accumulate raw and hard-NMS predictions and compute localization AP."""

    def __init__(
        self,
        annotations,
        ignored_videos,
        thresholds,
        nms_threshold,
        minimum_score,
        top_k,
    ):
        database = annotations["database"]
        videos = [
            video
            for video, item in database.items()
            if item["subset"] == "test" and video not in ignored_videos
        ]
        rows = [
            (video, annotation["segment"][0], annotation["segment"][1])
            for video in videos
            for annotation in database[video]["annotations"]
        ]
        self.ground_truth = pd.DataFrame(rows, columns=["video-id", "t-start", "t-end"])
        self.video_ids = set(videos)
        self.thresholds = list(thresholds)
        self.nms_threshold = nms_threshold
        self.minimum_score = minimum_score
        self.top_k = top_k
        self.predictions = {"raw": [], "nms": []}
        self.stats = {}
        print(
            f"{len(self.ground_truth)} ground truth instances from {len(videos)} videos"
        )

    def update(self, video, prediction):
        if video not in self.video_ids:
            return
        detections = np.column_stack(
            (
                prediction["segments"].detach().cpu().numpy(),
                prediction["scores"].detach().cpu().numpy(),
                prediction["labels"].detach().cpu().numpy(),
            )
        )
        for mode in self.predictions:
            selected = detections.copy()
            if mode == "nms":
                selected = apply_nms(
                    selected,
                    nms_threshold=self.nms_threshold,
                    minimum_score=self.minimum_score,
                    top_k=self.top_k,
                )
            selected = selected[selected[:, 2] > self.minimum_score]
            selected = selected[selected[:, 2].argsort()[::-1]][: self.top_k]
            selected[:, 3] = 0
            self.predictions[mode].extend(
                [[video, *detection] for detection in selected.tolist()]
            )

    def summarize(self):
        for mode, rows in self.predictions.items():
            table = pd.DataFrame(rows, columns=COLUMNS)
            self.predictions[mode] = table
            proposals = table.drop(columns="cls")
            per_iou = average_precision(
                self.ground_truth, proposals, np.asarray(self.thresholds)
            )
            self.stats[mode] = {
                "mAP": float(per_iou.mean()),
                "ap_values": [per_iou.tolist()],
                "per_iou_ap": per_iou.tolist(),
                "per_cls_ap": [float(per_iou.mean())],
                "AP50": float(per_iou[self.thresholds.index(0.5)]),
            }
            print(
                f"mode={mode} {len(table)} predictions from "
                f"{table['video-id'].nunique()} videos"
            )
            values = " ".join(f"{100 * value:.2f}" for value in per_iou)
            print(f"{values} {100 * per_iou.mean():.2f} {mode}")
