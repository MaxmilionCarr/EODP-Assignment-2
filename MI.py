import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils import resample
from scipy.stats import entropy
from preprocessing import quick_data

NORMAL_MI_THRESHOLD = 0.1  # threshold for selecting features

def MI_feature_ranks():
    X = quick_data().drop(columns=['persid'])
    y = X.pop('most_used_mode')

    le = LabelEncoder()
    y = le.fit_transform(y)

    n_boot = 5
    mi_scores = np.zeros((n_boot, X.shape[1]))

    for i in range(n_boot):
        print(f"Bootstrap iteration {i+1}/{n_boot}")
        Xb, yb = resample(X, y, replace=True, random_state=i)
        mi_scores[i] = mutual_info_classif(
            Xb, yb,
            discrete_features='auto',
            random_state=i,
        )

    mi_df = pd.DataFrame(mi_scores, columns=X.columns)
    mean_mi = mi_df.mean().sort_values(ascending=False)
    std_mi  = mi_df.std().reindex(mean_mi.index)

    # --- Entropy of the target (same for all features)
    p_y = np.bincount(y) / len(y)
    H_y = entropy(p_y, base=2)

    # --- Normalise each feature’s MI by sqrt(H_x * H_y)
    mean_mi_norm = pd.Series(index=mean_mi.index, dtype=float)
    std_mi_norm  = pd.Series(index=mean_mi.index, dtype=float)

    for col in mean_mi.index:
        x = X[col]
        # entropy of this feature’s distribution
        p_x = np.bincount(x) / len(x) if np.issubdtype(x.dtype, np.integer) \
              else np.bincount(LabelEncoder().fit_transform(x)) / len(x)
        H_x = entropy(p_x, base=2)
        if H_x > 0 and H_y > 0:
            mean_mi_norm[col] = mean_mi[col] / np.sqrt(H_x * H_y)
            std_mi_norm[col]  = std_mi[col]  / np.sqrt(H_x * H_y)
        else:
            mean_mi_norm[col] = 0.0
            std_mi_norm[col]  = 0.0

    print("Average normalised MI feature importance:\n", mean_mi_norm)
    print("\nStd of normalised MI feature importance:\n", std_mi_norm)

    selected_features = mean_mi_norm[mean_mi_norm > NORMAL_MI_THRESHOLD].index.tolist()
    print(f"\nSelected features (normalised MI > {NORMAL_MI_THRESHOLD}):\n", selected_features)

    return mean_mi, std_mi, mean_mi_norm, std_mi_norm, le.classes_



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





