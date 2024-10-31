import matplotlib.pyplot as plt
import numpy as np

def plot_results(data, indices, title, labels, line=True, scatter=False, bar=False):
    num_results = len(data)
    for i in range(num_results):
        unique_values, counts = np.unique(data[i], return_counts=True)
        if line:
            # plt.plot(indices, data[i], linestyle='-', linewidth=1, label=f'{labels[i]} (Line)')
            plt.plot(unique_values, counts, linestyle='-', linewidth=3, label=f'{labels[i]} (Line)')
        if scatter:
            # plt.scatter(indices, data[i], s=50, label=f'{labels[i]} (Dots)')
            plt.scatter(unique_values, counts, s=150, label=f'{labels[i]} (Dots)')
        if bar:
            width = 0.015
            x_shifted = unique_values + i*(width)
            plt.bar(x_shifted, counts, width=width, alpha=0.7, label=f'{labels[i]}')

    plt.xlabel('Achieved Accuracy Scores')
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(left=-0.02)  # X-axis starts from 0
    plt.ylim(bottom=-1)  # Y-axis starts from -1


def compareNplot(precisions, recalls, f1_scores, all_error_cases, labels=None, model='llama3.1-8B'):
    """
    all_error_cases (dict of dicts) : 각 label에 해당하는 세팅에서 진행한 결과의 에러케이스들을 담은 딕셔너리들로 이뤄진 딕셔너리, 각 키값은 test에 상응한 라벨값이고 밸류값은 그 케이스에서 분류된 에러케이스들에 대한 딕셔너리임
    """
    assert len(precisions) == len(recalls) == len(f1_scores)
    for (p, r, f) in zip(precisions, recalls, f1_scores):
        assert len(p) == len(r) == len(f)    

    # Labels and categories
    if labels is None:
        labels = [
            'Single-Turn : Zero-Shot + Schemas', 
            'Multi-Turn : Zero-Shot + Schemas', 
            'Single-Turn : One-Shot (1) + Schemas', 
            'Multi-Turn : One-Shot (1) + Schemas',
            'Single-Turn : Few-Shot (10) + Schemas', 
        ]

    criterias = ['Precision', 'Recall', 'F1']

    # Prepare for visualization
    indices = range(len(precisions[0]))  # Assuming all lists have the same length

    # Plot each category (3: Hist Plot)
    plt.figure(figsize=(20, 30))  # each subplots will have 150 * 40 space
    for i, (data, criteria) in enumerate(zip( 
            [precisions, recalls, f1_scores],  # data
            criterias  # criteria
        )):
        plt.subplot(3, 1, i + 1)
        plot_results(data, indices, criteria, labels, line=False, scatter=False, bar=True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"./Logs/Extraction_Quality/({model}).png")
    plt.close()

    # Plot Pie Chart for each Test's Error Cases
    colors = [
        '#8085ff',  # First (light purple)
        '#ff7575',  # Second (vibrant peach)
        '#c2c2c2',  # Third (neutral grey)
        '#80b3ff',  # Fourth (muted blue)
        '#ffb3b3',  # Fifth (lighter peach)
        '#80e0ff',  # Sixth (brighter blue)
        '#ffb3ce',  # Seventh (weak pink)
        '#ffcc66'   # Eighth (yellow)
    ]
    for label in labels:
        error_data = all_error_cases[label]
        error_types = list(error_data.keys())
        counts = list(error_data.values())
        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(counts, autopct='%1.1f%%', startangle=90, counterclock=False, colors=colors)
        plt.title(f'Error Type Dist for {label} on {model}')
        plt.legend(error_types, title="Error Types", loc="best")
        plt.axis('equal')
        plt.savefig(f"./Logs/Error_Analysis/({label})-({model}).png")
        plt.close()
    """
    # Plot each category (1: Line Plot)
    plt.figure(figsize=(20, 30))  # each subplots will have 150 * 40 space
    for i, (data, criteria) in enumerate(zip( 
            [precisions, recalls, f1_scores],  # data
            criterias  # criteria
        )):
        plt.subplot(3, 1, i + 1)
        plot_results(data, indices, criteria, labels, line=True, scatter=False)
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"./Logs/Extraction_Quality/({model}).png")
    plt.close()

    # Plot each category (2: Scatter Plot)
    plt.figure(figsize=(20, 30))  # each subplots will have 150 * 40 space
    for i, (data, criteria) in enumerate(zip( 
            [precisions, recalls, f1_scores],  # data
            criterias  # criteria
        )):
        plt.subplot(3, 1, i + 1)
        plot_results(data, indices, criteria, labels, line=False, scatter=True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"./Logs/plot(scatter)({model}).png")
    plt.close()
    """
def errorCase_analysis(gold_list, extracted_list):
    """
    gold_list (list of list): Gold Extraction 내용이 케이스별로 리스트에 담겨있음
    extracted_list (list of list): 실제 Extract한 내용이 케이스별로 리스트에 담겨있음
    """
    assert len(gold_list) == len(extracted_list)
    