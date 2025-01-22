from sklearn import metrics as skm
from netcal.presentation import ReliabilityDiagram
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from netcal.metrics import ECE


def make_bins(confidences, is_corrected, bins):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_sizes = []
    bin_corrects = []

    for i in range(bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        bin_is_corrected_value = is_corrected[in_bin]
        bin_sizes.append(np.sum(in_bin))
        bin_corrects.append(np.sum(bin_is_corrected_value))

    return bin_sizes, bin_corrects


class SimpleStatsCache:
    def __init__(self, confids, correct):
        self.confids = np.array(confids)
        self.correct = np.array(correct)

    @property
    def rc_curve_stats(self):
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights

    @property
    def residuals(self):
        return 1 - self.correct


def area_under_risk_coverage_score(confids, correct):
    stats_cache = SimpleStatsCache(confids, correct)
    _, risks, weights = stats_cache.rc_curve_stats
    AURC_DISPLAY_SCALE = 1000
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]) * AURC_DISPLAY_SCALE


def compute_conf_metrics(y_true, y_confs):

    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    # print("accuracy: ", accuracy)
    result_matrics['acc'] = accuracy

    # use np to test if y_confs are all in [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)

    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    # print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    # print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1 - y_true, 1 - y_confs)
    # print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc

    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    aurc = area_under_risk_coverage_score(y_confs, y_true)
    result_matrics['aurc'] = aurc
    # print("AURC score:", aurc)

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    # print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics


def visualization(bin_sizes, bin_corrects, save_path, acc, ece, auroc, bins):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    bin_boundaries = np.linspace(0, 100, bins + 1)
    x = bin_boundaries[1:]

    # First visualization
    bin_wrongs = np.array(bin_sizes) - np.array(bin_corrects)
    axes[0].bar(x, bin_corrects, width=4,
                label='Correct Answers', edgecolor='black', color="#4C84FF")
    axes[0].bar(x, bin_wrongs, width=4,
                label='Wrong Answers', edgecolor='black', color='#FF4C4C')
    axes[0].set_xlabel('Confidence (%)', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title(
        f'ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}', fontweight='bold')
    axes[0].set_xticks(bin_boundaries)
    axes[0].set_yticks([])
    axes[0].set_xticklabels([f'{int(b)}%' for b in bin_boundaries])
    axes[0].legend()

    # Second visualization
    bin_boundaries = np.linspace(0, 100, bins + 1)
    bin_correct_ratios = [
        correct / size if size > 0 else 0
        for correct, size in zip(bin_corrects, bin_sizes)
    ]
    axes[1].bar(x, bin_correct_ratios, width=4,
                label='Correct Answers', edgecolor='black', color="#4C84FF")
    axes[1].plot([0, 100], [0, 1], linestyle='--',
                 color='black', label='y = x Line')
    axes[1].set_xlim(0, 105)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel('Confidence', fontweight='bold')
    axes[1].set_ylabel('Accuracy within Bin', fontweight='bold')
    axes[1].set_xticks(bin_boundaries)
    axes[1].set_yticks([])
    axes[1].set_xticklabels([f'{b/100:.1f}' for b in bin_boundaries])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)


def load_csv_file(csv_path):
    df = pd.read_csv(csv_path)

    df['confidence_score'] = df['confidence_score'].fillna(0)
    df['normalized_confidence'] = (df['confidence_score'] - df['confidence_score'].min()) / \
        (df['confidence_score'].max() - df['confidence_score'].min())

    df['is_correct'] = df['is_correct'].astype(int)
    is_correct_list = np.array(df['is_correct'].tolist())
    confidence_list = np.array(df['normalized_confidence'].tolist())

    return confidence_list, is_correct_list


results_csvdir_path = "results_csv"
output_dir = "outputs/visualizations"
os.makedirs(output_dir, exist_ok=True)


def process_csv_files(directory, output_dir):
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            csv_path = os.path.join(directory, file_name)
            save_path = os.path.join(
                output_dir, f"vis1_{os.path.splitext(file_name)[0]}.png")

            bins = 10
            confidences, is_corrected = load_csv_file(csv_path)
            bin_sizes, bin_corrects = make_bins(
                confidences, is_corrected, bins=bins)
            results = compute_conf_metrics(is_corrected, confidences)
            visualization(bin_sizes, bin_corrects,
                          save_path, results['acc'], results['ece'], results['auroc'], bins=bins)
            print(f"Processed and saved visualization for {file_name}")


process_csv_files(results_csvdir_path, output_dir)
