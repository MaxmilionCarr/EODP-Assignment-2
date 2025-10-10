import numpy as np
import pandas as pd
from scipy.stats import entropy
from preprocessing import quick_data
import matplotlib.pyplot as plt
import os

GRAPH_DIR = "MI_graphs"
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def compute_prob(labels: pd.Series, weights: pd.Series = None):
    '''
    Return probability of each category in `labels`, weighted if weights given.
    '''
    if weights is None:
        counts = labels.value_counts()
        return counts / counts.sum()
    else:
        w = pd.Series(weights, index=labels.index)
        # total weight per category / total weight
        return w.groupby(labels).sum() / w.sum()

def compute_entropy(labels: pd.Series, weights: pd.Series = None):
    '''
    Compute entropy (base 2) of a discrete variable, weighted if weights given.
    '''
    p = compute_prob(labels, weights)
    return entropy(p, base=2)

def compute_conditional_entropy(x: pd.Series, y: pd.Series, weights: pd.Series = None):
    ''''
    Compute H(Y|X), the conditional entropy of Y given X, weighted if weights given.
    AI Acknowledgement, ChatGPT was used to help incorporate weightings for entropy calculations
    in the function below. (Too many prompts to list here)
    '''
    w = pd.Series(weights, index=x.index) if weights is not None else None
    p_x = compute_prob(x, w)

    # weighted entropy of Y inside each X group
    temp = pd.DataFrame({'X': x, 'Y': y})
    if w is not None:
        temp['w'] = w
        def group_entropy(g):
            return compute_entropy(g['Y'], g['w'])
    else:
        def group_entropy(g):
            return compute_entropy(g['Y'])

    H_y_given_x = temp.groupby('X').apply(group_entropy)
    return (p_x * H_y_given_x).sum()

def compute_normalized_mutual_info(x: pd.Series, y: pd.Series, weights: pd.Series = None):
    '''
    Symmetric NMI = I(X;Y)/sqrt(H(X)H(Y)), weighted if weights given.
    '''
    H_x = compute_entropy(x, weights)
    H_y = compute_entropy(y, weights)
    H_y_given_x = compute_conditional_entropy(x, y, weights)
    return (H_y - H_y_given_x) / np.sqrt(H_x * H_y) if H_x > 0 and H_y > 0 else 0.0

def main():
    '''
    Compute and plot NMI of all features with target variable.
    '''
    df = quick_data().drop(columns=['persid'])
    y = df.pop('most_used_mode')
    weights = df.pop('perspoststratweight')

    # Compute NMI for each feature
    nmi = pd.DataFrame(index=df.columns, columns=['NMI'])
    for col in df.columns:
        nmi.loc[col] = compute_normalized_mutual_info(df[col], y, weights)

    print(nmi.sort_values(by='NMI', ascending=False))

    # Plot NMI values
    nmi.sort_values(by='NMI', ascending=True).plot.barh(figsize=(10, 8), legend=False)
    plt.xlabel('Normalized Mutual Information (NMI)')
    plt.title('Feature Selection using NMI')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "nmi_feature_selection.png"), bbox_inches="tight")
    plt.show() 

if __name__ == "__main__":
    main()