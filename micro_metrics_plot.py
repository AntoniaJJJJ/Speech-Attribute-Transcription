import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Fixed list of classes
CLASSES = [
    'n_anterior', 'n_back', 'n_consonantal', 'n_continuant', 'n_coronal',
    'n_delayedrelease', 'n_diphthong', 'n_distributed', 'n_front', 'n_high',
    'n_labial', 'n_lateral', 'n_long', 'n_low', 'n_monophthong', 'n_nasal',
    'n_offglide', 'n_onglide', 'n_round', 'n_short', 'n_sibilant',
    'n_sonorant', 'n_syllabic', 'n_tense', 'n_voice', 'p_anterior', 'p_back',
    'p_consonantal', 'p_continuant', 'p_coronal', 'p_diphthong', 'p_distributed',
    'p_front', 'p_high', 'p_labial', 'p_lateral', 'p_long', 'p_low',
    'p_monophthong', 'p_nasal', 'p_offglide', 'p_onglide', 'p_round',
    'p_short', 'p_sibilant', 'p_sonorant', 'p_tense', 'p_voice'
]

def bar_plot(precision, recall, f1):
    """Generate a bar plot for PRE, REC, and F1."""
    x = np.arange(len(CLASSES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1 Score')
    ax.set_xlabel('Classes')
    ax.set_title('Precision, Recall, and F1 by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()

def heatmap(precision, recall, f1):
    """Generate a heatmap for PRE, REC, and F1."""
    data = np.array([precision, recall, f1])
    sns.heatmap(data, annot=True, cmap="YlGnBu", xticklabels=CLASSES, yticklabels=['Precision', 'Recall', 'F1'])
    plt.title('Precision, Recall, and F1 Heatmap')
    plt.tight_layout()
    plt.show()

def line_plot(precision, recall, f1):
    """Generate a line plot for PRE, REC, and F1."""
    plt.figure(figsize=(12, 6))
    plt.plot(CLASSES, precision, marker='o', label='Precision')
    plt.plot(CLASSES, recall, marker='o', label='Recall')
    plt.plot(CLASSES, f1, marker='o', label='F1 Score')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Trends by Class')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def radar_plot(precision, recall, f1):
    """Generate a radar plot for PRE, REC, and F1."""
    metrics = ['Precision', 'Recall', 'F1']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, class_name in enumerate(CLASSES[:5]):  # Limit to first 5 classes for readability
        data = [precision[i], recall[i], f1[i]]
        data += data[:1]
        ax.plot(angles, data, label=class_name)
        ax.fill(angles, data, alpha=0.25)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Radar Chart for PRE, REC, and F1 (Sample Classes)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
    plt.tight_layout()
    plt.show()

def precision_recall_curve_plot(y_true, y_scores):
    """Generate a precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def main(plot_type, precision, recall, f1, y_true=None, y_scores=None):
    """Main function to generate the requested plot."""
    if plot_type == "bar":
        bar_plot(precision, recall, f1)
    elif plot_type == "heatmap":
        heatmap(precision, recall, f1)
    elif plot_type == "line":
        line_plot(precision, recall, f1)
    elif plot_type == "radar":
        radar_plot(precision, recall, f1)
    elif plot_type == "pr_curve":
        if y_true is None or y_scores is None:
            raise ValueError("y_true and y_scores are required for precision-recall curve.")
        precision_recall_curve_plot(y_true, y_scores)
    else:
        raise ValueError(f"Invalid plot type: {plot_type}. Choose from bar, heatmap, line, radar, pr_curve.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a plot for Precision, Recall, and F1 analysis.")
    parser.add_argument("plot_type", choices=["bar", "heatmap", "line", "radar", "pr_curve"], help="Type of plot to generate.")
    parser.add_argument("--precision", nargs="+", type=float, required=True, help="Precision values for each class.")
    parser.add_argument("--recall", nargs="+", type=float, required=True, help="Recall values for each class.")
    parser.add_argument("--f1", nargs="+", type=float, required=True, help="F1 values for each class.")
    parser.add_argument("--y_true", nargs="+", type=int, help="True labels for precision-recall curve.")
    parser.add_argument("--y_scores", nargs="+", type=float, help="Predicted scores for precision-recall curve.")

    args = parser.parse_args()
    main(args.plot_type, args.precision, args.recall, args.f1, args.y_true, args.y_scores)
