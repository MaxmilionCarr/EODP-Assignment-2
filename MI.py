import numpy as np
import pandas as pd
from scipy.stats import entropy
from preprocessing import quick_data

NORMAL_MI_THRESHOLD = 0.05  # threshold for selecting features


def compute_prob(labels: pd.Series, weights: pd.Series = None):
    """Return probability of each category in `labels`, weighted if weights given."""
    if weights is None:
        counts = labels.value_counts()
        return counts / counts.sum()
    else:
        w = pd.Series(weights, index=labels.index)
        # total weight per category ÷ total weight
        return w.groupby(labels).sum() / w.sum()

def compute_entropy(labels: pd.Series, weights: pd.Series = None):
    """Entropy (base 2) of a discrete variable, weighted if weights given."""
    p = compute_prob(labels, weights)
    return entropy(p, base=2)

def compute_conditional_entropy(x: pd.Series, y: pd.Series, weights: pd.Series = None):
    """Conditional entropy H(Y|X), weighted if weights given."""
    w = pd.Series(weights, index=x.index) if weights is not None else None
    # total weight (or count) of each X category
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
    """Symmetric NMI = I(X;Y)/sqrt(H(X)H(Y)), weighted if weights given."""
    H_x = compute_entropy(x, weights)
    H_y = compute_entropy(y, weights)
    H_y_given_x = compute_conditional_entropy(x, y, weights)
    return (H_y - H_y_given_x) / np.sqrt(H_x * H_y) if H_x > 0 and H_y > 0 else 0.0

def select_features(df: pd.DataFrame, target_col='most_used_mode', weights_col='perspoststratweight', threshold=NORMAL_MI_THRESHOLD):
    """Select features with NMI above threshold."""
    y = df[target_col]
    weights = df[weights_col] if weights_col in df.columns else None
    features = df.drop(columns=[target_col, weights_col] if weights_col in df.columns else [target_col])

    selected_features = []
    for col in features.columns:
        nmi = compute_normalized_mutual_info(features[col], y, weights)
        if nmi >= threshold:
            selected_features.append(col)
            print(f"Selected {col} with NMI={nmi:.4f}")
        else:
            print(f"Rejected {col} with NMI={nmi:.4f}")

    return selected_features

if __name__ == "__main__":
    df = quick_data().drop(columns=['persid'])
    training_len = int(len(df) * 0.6)
    training = df[0:training_len]
    testing  = df[training_len:]
    y = df.pop('most_used_mode')
    weights = df.pop('perspoststratweight')

    nmi = pd.DataFrame(index=df.columns, columns=['NMI'])
    for col in df.columns:
        nmi.loc[col] = compute_normalized_mutual_info(df[col], y, weights)

    print(nmi.sort_values(by='NMI', ascending=False))





