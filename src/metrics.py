import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def calculate_ece(confidences, is_corrected, bins):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_sizes = []
    bin_corrects = []
    ece = 0

    for i in range(bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if np.any(in_bin):
            bin_confidences_value = confidences[in_bin]
            bin_is_corrected_value = is_corrected[in_bin]
            bin_accuracy = np.mean(bin_is_corrected_value)
            bin_confidence = np.mean(bin_confidences_value)
            bin_size = np.mean(in_bin)
            ece += bin_size * abs(bin_confidence - bin_accuracy)

        bin_sizes.append(np.sum(in_bin))
        bin_corrects.append(np.sum(bin_is_corrected_value))

    return ece, bin_sizes, bin_corrects


def visualization_first(bin_sizes, bin_corrects, save_path, acc, ece, auroc, bins):
    plt.figure(figsize=(10, 6))
    bin_boundaries = np.linspace(0, 100, bins + 1)
    x = bin_boundaries[1:]
    bin_wrongs = np.array(bin_sizes) - np.array(bin_corrects)

    plt.bar(x, bin_corrects, width=4,
            label='Correct Answers', edgecolor='black')
    plt.bar(x, bin_wrongs, width=4,
            label='Wrong Answers', edgecolor='black')

    plt.xlabel('Confidence (%)', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title(
        f'ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}', fontweight='bold')
    plt.xticks(bin_boundaries, [
               f'{int(b)}%' for b in bin_boundaries])
    plt.yticks([])
    plt.legend()

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

            confidences, is_corrected = load_csv_file(csv_path)
            acc = np.mean(is_corrected)
            ece, bin_sizes, bin_corrects = calculate_ece(
                confidences, is_corrected, bins=10)

            try:
                auroc = roc_auc_score(is_corrected, confidences)
            except:
                auroc = 0.0
            visualization_first(bin_sizes, bin_corrects,
                                save_path, acc, ece, auroc, bins=10)
            print(f"Processed and saved visualization for {file_name}")


process_csv_files(results_csvdir_path, output_dir)
