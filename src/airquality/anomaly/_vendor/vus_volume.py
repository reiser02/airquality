"""VUS-ROC / VUS-PR (Volume Under the Surface) computation.

Trimmed copy of the ``metricor`` class from VUS 0.0.6
(``vus/utils/metrics.py``, The DATUM Lab, Apache-2.0). Only the methods needed
to reproduce ``generate_curve``'s ``avg_auc_3d`` (VUS_ROC) and ``avg_ap_3d``
(VUS_PR) are kept: ``range_convers_new``, ``new_sequence``, ``sequencing`` and
``RangeAUC_volume_opt``. The methods are copied from upstream so results match
the original ``vus.metrics.get_metrics`` exactly; the only local change is
hoisting the per-threshold prediction masks out of the window loop in
``RangeAUC_volume_opt`` (identical arrays, identical results, O(windowSize)
times fewer full-series comparisons). See ``NOTICE`` for attribution.
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


def vus_roc_pr(labels, score, sliding_window, thre=250):
    """Return ``(VUS_ROC, VUS_PR)`` for the given labels/score.

    Mirrors ``vus.analysis.robustness_eval.generate_curve(...)[-2:]`` with the
    default ``version='opt'`` path.
    """
    *_, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(
        labels_original=labels, score=score, windowSize=sliding_window, thre=thre
    )
    return avg_auc_3d, avg_ap_3d
