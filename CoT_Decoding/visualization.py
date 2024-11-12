import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
def load_evaluation_results(file_path):
    """
    Load the evaluation results from a CSV file.
    """
    return pd.read_csv(file_path)

# Plot histograms of confidence scores
def plot_confidence_histograms(data, description):
    """
    Plot histograms of confidence scores for correct and incorrect predictions and save them as files.
    """
    correct_predictions = data[data['is_correct'] == True]
    incorrect_predictions = data[data['is_correct'] == False]

    plt.figure(figsize=(14, 6))
    
    # Histogram for correct predictions
    plt.subplot(1, 2, 1)
    plt.hist(correct_predictions['confidence_score'], bins=20, alpha=0.7, color='g')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'{description} - Confidence Histogram for Correct Predictions')

    # Histogram for incorrect predictions
    plt.subplot(1, 2, 2)
    plt.hist(incorrect_predictions['confidence_score'], bins=20, alpha=0.7, color='r')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'{description} - Confidence Histogram for Incorrect Predictions')

    plt.tight_layout()
    plt.savefig(f'{description}_confidence_histograms.png')
    plt.close()

# Plot comparison of confidence scores
def plot_confidence_comparison(data, description):
    """
    Plot a chart comparing confidence scores for correct and incorrect predictions and save it as a file.
    """
    correct_predictions = data[data['is_correct'] == True]
    incorrect_predictions = data[data['is_correct'] == False]

    plt.figure(figsize=(10, 6))
    
    plt.hist(correct_predictions['confidence_score'], bins=20, alpha=0.5, label='Correct Predictions', color='g')
    plt.hist(incorrect_predictions['confidence_score'], bins=20, alpha=0.5, label='Incorrect Predictions', color='r')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'{description} - Confidence Score Comparison')
    plt.legend()
    plt.savefig(f'{description}_confidence_comparison.png')
    plt.close()

# Plot difference between predicted and correct answer for incorrect predictions
def plot_difference_in_predictions(data, description):
    """
    Plot the difference between the predicted final answer and correct answer for incorrect predictions and save it as a file.
    """
    incorrect_predictions = data[data['is_correct'] == False].copy()

    # Clean the predicted_final_answer and correct_answer columns, ensuring numeric conversion
    incorrect_predictions['predicted_final_answer'] = pd.to_numeric(incorrect_predictions['predicted_final_answer'], errors='coerce')
    incorrect_predictions['correct_answer'] = pd.to_numeric(incorrect_predictions['correct_answer'], errors='coerce')

    # Drop rows where conversion failed
    incorrect_predictions = incorrect_predictions.dropna(subset=['predicted_final_answer', 'correct_answer'])

    # Calculate the difference between predicted and correct answers
    incorrect_predictions['difference'] = incorrect_predictions['predicted_final_answer'] - incorrect_predictions['correct_answer']

    plt.figure(figsize=(10, 6))
    plt.hist(incorrect_predictions['difference'], bins=20, alpha=0.7, color='b')
    plt.xlabel('Difference (Predicted - Correct)')
    plt.ylabel('Frequency')
    plt.title(f'{description} - Difference in Predicted and Correct Answer for Incorrect Predictions')
    plt.savefig(f'{description}_difference_in_predictions.png')
    plt.close()

if __name__ == '__main__':
    # Load MultiArith results
    multiarith_data = load_evaluation_results('results/MultiArith_Baseline_llama3-1-8b_k10_AggFalse_evaluation_results.csv')
    plot_confidence_histograms(multiarith_data, 'MultiArith')
    plot_confidence_comparison(multiarith_data, 'MultiArith')
    plot_difference_in_predictions(multiarith_data, 'MultiArith')

    # Load GSM8K results
    gsm8k_data = load_evaluation_results('results/GSM8K_Baseline_llama3-1-8b_k10_AggFalse_evaluation_results.csv')
    plot_confidence_histograms(gsm8k_data, 'GSM8K')
    plot_confidence_comparison(gsm8k_data, 'GSM8K')
    plot_difference_in_predictions(gsm8k_data, 'GSM8K')
