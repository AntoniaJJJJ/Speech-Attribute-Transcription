"""
Multilabel Classification Metrics for Speech Recognition Dataset

Description:
This script processes a Hugging Face `datasets` dataset containing multilabel classification results for 
a speech recognition task. The dataset is expected to include two critical columns:
    - `target_text`: The ground truth labels for each example, represented as lists of phoneme attributes.
    - `pred_str`: The predicted labels for each example, also represented as lists of phoneme attributes.

Metrics Computed:
1. **Micro Metrics (Per Class)**:
    - Precision, Recall, and F1 scores are calculated for each class (phoneme attribute) individually.
    - These metrics help identify the performance of the system on specific phoneme attributes.

2. **Macro Metrics (Overall)**:
    - Precision, Recall, and F1 scores are averaged across all classes, treating all classes equally.
    - These metrics summarize the overall performance of the system.


Example Output:
Number of unique classes: 50

Micro Metrics (Per Class):
Class 'n_high': Precision=0.850, Recall=0.820, F1=0.835
Class 'p_high': Precision=0.780, Recall=0.800, F1=0.790
Class 'n_low': Precision=0.890, Recall=0.860, F1=0.875

Overall Metrics (Macro-Averaged):
Precision: 0.840
Recall: 0.827
F1 Score: 0.833



"""
def get_unique_classes(target_texts):
    """Extract all unique phoneme attribute classes from the target_text column."""
    unique_classes = set()
    for example in target_texts:
        for phoneme_labels in example:
            # Split space-separated attributes in each string
            attributes = phoneme_labels.split()
            unique_classes.update(attributes)
    return sorted(unique_classes)


def convert_to_binary_matrix(data, unique_classes):
    """
    Convert the multilabel sequences into a binary matrix for metric calculations.
    Each unique class becomes a column, and each row represents the labels for a phoneme.
    """
    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}
    binary_matrix = []
    for sequence in data:
        for phoneme_labels in sequence:
            # Split space-separated attributes in each string
            attributes = phoneme_labels.split()
            row = [0] * len(unique_classes)
            for label in attributes:
                if label not in class_to_index:
                    raise ValueError(f"Unexpected label '{label}' in data.")
                row[class_to_index[label]] = 1
            binary_matrix.append(row)
    return binary_matrix, class_to_index


def calculate_metrics(y_true, y_pred, unique_classes):
    """
    Calculate micro-averaged and macro-averaged metrics.
    """
    # Compute micro metrics per class
    pre_class, rec_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Compute macro-averaged metrics
    pre_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return pre_class, rec_class, f1_class, pre_macro, rec_macro, f1_macro


def process_dataset(dataset_path):
    """Load the dataset and compute metrics."""
    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Ensure the dataset has 'target_text' and 'pred_str' columns
    if 'target_text' not in dataset.column_names or 'pred_str' not in dataset.column_names:
        raise ValueError("Dataset must contain 'target_text' and 'pred_str' columns.")

    target_texts = dataset['target_text']
    pred_strs = dataset['pred_str']

    # Extract unique classes
    unique_classes = get_unique_classes(target_texts)
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Classes: {unique_classes}")

    # Convert target_text and pred_str to binary matrices
    y_true, class_to_index = convert_to_binary_matrix(target_texts, unique_classes)
    y_pred, _ = convert_to_binary_matrix(pred_strs, unique_classes)

    # Compute metrics
    pre_class, rec_class, f1_class, pre_macro, rec_macro, f1_macro = calculate_metrics(y_true, y_pred, unique_classes)

    # Display micro metrics for each class
    print("\nMicro Metrics (Per Class):")
    for cls, idx in class_to_index.items():
        print(f"Class '{cls}': Precision={pre_class[idx]:.3f}, Recall={rec_class[idx]:.3f}, F1={f1_class[idx]:.3f}")

    # Display macro metrics
    print("\nOverall Metrics (Macro-Averaged):")
    print(f"Precision: {pre_macro:.3f}")
    print(f"Recall: {rec_macro:.3f}")
    print(f"F1 Score: {f1_macro:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Precision, Recall, and F1 scores for multilabel speech recognition results.")
    parser.add_argument("dataset_path", help="Path to the folder containing the Hugging Face dataset.")
    args = parser.parse_args()

    process_dataset(args.dataset_path)