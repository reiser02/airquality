"""VUS-ROC / VUS-PR (Volume Under the Surface) computation.

Trimmed copy of the ``metricor`` class from VUS 0.0.6
(``vus/utils/metrics.py``, The DATUM Lab, Apache-2.0). The single-series path
matches upstream exactly. Local changes hoist threshold masks out of the window
loop and add a segmented path that pools statistics while preventing events
from crossing sequence bounds. See ``NOTICE`` for attribution.
"""

from __future__ import annotations

import numpy as np


class metricor:
    def range_convers_new(self, label):
        """
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        """
        L = []
        i = 0
        j = 0
        while j < len(label):
            while label[i] == 0:
                i += 1
                if i >= len(label):
                    break
            j = i + 1
            if j >= len(label):
                if j == len(label):
                    L.append((i, j - 1))
                break
            while label[j] != 0:
                j += 1
                if j >= len(label):
                    L.append((i, j - 1))
                    break
            if j >= len(label):
                break
            L.append((i, j - 1))
            i = j
        return L

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        # The per-threshold prediction masks do not depend on `window`, so they
        # are computed once here instead of `thre` times per window level (the
        # upstream code recomputed `score >= threshold` inside the window loop).
        # Same arrays, same order: results are bit-identical to the original.
        threshold_positions = np.linspace(0, len(score) - 1, thre).astype(int)
        pred_masks = [score >= score_sorted[i] for i in threshold_positions]

        for k, pred in enumerate(pred_masks):
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for pred in pred_masks:
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

    def RangeAUC_volume_opt_segments(
        self, labels_by_segment, scores_by_segment, windowSize, thre=250
    ):
        """Pool VUS statistics across independent temporal sequences."""
        labels_by_segment = [np.asarray(labels).ravel() for labels in labels_by_segment]
        scores_by_segment = [
            np.asarray(scores, dtype=float).ravel() for scores in scores_by_segment
        ]
        if not labels_by_segment or len(labels_by_segment) != len(scores_by_segment):
            raise ValueError("VUS requires matching non-empty segment lists")
        if any(
            labels.shape != scores.shape or labels.size == 0
            for labels, scores in zip(labels_by_segment, scores_by_segment, strict=True)
        ):
            raise ValueError("VUS segment labels and scores must have matching non-empty shapes")
        if any(not np.isin(labels, (0, 1)).all() for labels in labels_by_segment):
            raise ValueError("VUS segment labels must be binary")
        if any(not np.isfinite(scores).all() for scores in scores_by_segment):
            raise ValueError("VUS segment scores must be finite")
        if windowSize < 0 or thre < 1:
            raise ValueError("VUS windowSize must be non-negative and thre must be positive")

        score = np.concatenate(scores_by_segment)
        n_points = len(score)
        positives = float(sum(np.sum(labels) for labels in labels_by_segment))
        if positives == 0 or positives == n_points:
            raise ValueError("VUS requires both classes across all segments")
        if len(labels_by_segment) == 1:
            return self.RangeAUC_volume_opt(
                labels_by_segment[0], scores_by_segment[0], windowSize, thre
            )

        sequences = [self.range_convers_new(labels) for labels in labels_by_segment]
        maximum_ranges = [
            self.new_sequence(labels, sequence, windowSize) if sequence else []
            for labels, sequence in zip(labels_by_segment, sequences, strict=True)
        ]
        score_sorted = -np.sort(-score)
        threshold_positions = np.linspace(0, n_points - 1, thre).astype(int)
        flat_masks = [score >= score_sorted[index] for index in threshold_positions]
        split_points = np.cumsum([len(scores) for scores in scores_by_segment])[:-1]
        pred_masks = [np.split(mask, split_points) for mask in flat_masks]

        window_3d = np.arange(0, windowSize + 1, 1)
        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))
        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        for window in window_3d:
            extended = [
                self.sequencing(labels, sequence, window)
                if sequence
                else labels.astype(float)
                for labels, sequence in zip(labels_by_segment, sequences, strict=True)
            ]
            ranges = [
                self.new_sequence(labels, sequence, window) if sequence else []
                for labels, sequence in zip(extended, sequences, strict=True)
            ]
            total_ranges = sum(len(value) for value in ranges)
            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)

            for threshold_index, masks_by_segment in enumerate(pred_masks, start=1):
                true_positives = 0.0
                weighted_labels = 0.0
                existence = 0
                predicted = float(sum(np.sum(mask) for mask in masks_by_segment))

                for labels, sequence, local_ranges, max_ranges, mask in zip(
                    extended,
                    sequences,
                    ranges,
                    maximum_ranges,
                    masks_by_segment,
                    strict=True,
                ):
                    if not sequence:
                        continue
                    weighted = labels.copy()
                    for start, end in local_ranges:
                        local = slice(start, end + 1)
                        weighted[local] = labels[local] * mask[local]
                        existence += int(mask[local].any())
                    for start, end in sequence:
                        weighted[start : end + 1] = 1
                    for start, end in max_ranges:
                        local = slice(start, end + 1)
                        true_positives += float(np.dot(weighted[local], mask[local]))
                        weighted_labels += float(np.sum(weighted[local]))

                adjusted_positives = (positives + weighted_labels) / 2.0
                recall = min(true_positives / adjusted_positives, 1.0)
                tpr = recall * (existence / total_ranges)
                false_positives = predicted - true_positives
                fpr = false_positives / (n_points - adjusted_positives)
                precision = true_positives / predicted
                TF_list[threshold_index] = [tpr, fpr]
                Precision_list[threshold_index] = precision

            TF_list[thre + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list
            auc_3d[window] = np.dot(
                TF_list[1:, 1] - TF_list[:-1, 1],
                (TF_list[1:, 0] + TF_list[:-1, 0]) / 2,
            )
            ap_3d[window] = np.dot(
                TF_list[1:-1, 0] - TF_list[:-2, 0], Precision_list[1:]
            )

        return (
            tpr_3d,
            fpr_3d,
            prec_3d,
            window_3d,
            float(np.mean(auc_3d)),
            float(np.mean(ap_3d)),
        )


def vus_roc_pr(labels, score, sliding_window, thre=250):
    """Return ``(VUS_ROC, VUS_PR)`` for the given labels/score.

    Mirrors ``vus.analysis.robustness_eval.generate_curve(...)[-2:]`` with the
    default ``version='opt'`` path.
    """
    *_, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(
        labels_original=labels, score=score, windowSize=sliding_window, thre=thre
    )
    return avg_auc_3d, avg_ap_3d


def vus_roc_pr_segments(labels_by_segment, scores_by_segment, sliding_window, thre=250):
    """Return VUS pooled across independent temporal sequences."""
    *_, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt_segments(
        labels_by_segment, scores_by_segment, sliding_window, thre
    )
    return avg_auc_3d, avg_ap_3d
