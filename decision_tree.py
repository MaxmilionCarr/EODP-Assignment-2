import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from preprocessing import quick_data
from MI import select_features


def evaluate_single_bootstrap(X, y, weights, depth, min_samples, seed):
    """Train + evaluate one bootstrap split of the decision tree."""
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=seed
    )

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        min_samples_leaf=min_samples,
        random_state=seed
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
    return acc, (depth, min_samples)


def run_decision_tree(selected_features, target_col='most_used_mode', random_state=42, n_boot=50, n_jobs=-1):
    df = quick_data().drop(columns=['persid'])

    # Group transport modes
    mapping = {
        "Bicycle": "Active", "Mobility Scooter": "Active", "Motorcycle": "Private",
        "Public Bus": "Public", "Rideshare Service": "Public", "School Bus": "Public",
        "Taxi": "Private", "Train": "Public", "Tram": "Public",
        "Vehicle Driver": "Private", "Vehicle Passenger": "Private",
        "Walking": "Active", "Other": "Private",
        "Plane": "Public", "Running/jogging": "Active"
    }
    df["most_used_mode"] = df["most_used_mode"].replace(mapping)

    weights = df['perspoststratweight'] if 'perspoststratweight' in df.columns else None
    X = df[selected_features]
    y = df[target_col]

    # store accuracy + params across bootstraps
    all_results = []

    for depth in range(1, 10):
        for min_samples in range(1, 6):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_single_bootstrap)(X, y, weights, depth, min_samples, random_state+b)
                for b in range(n_boot)
            )
            for acc, params in results:
                all_results.append((acc, params))

    # ---- Aggregate results ----
    results_df = pd.DataFrame(all_results, columns=["accuracy", "params"])
    mean_acc = results_df.groupby("params")["accuracy"].mean().sort_values(ascending=False)
    best_params = mean_acc.index[0]
    print("\nMean accuracy by params:\n", mean_acc)
    print("\nBest params on average:", best_params, "with mean accuracy:", mean_acc.iloc[0])

    # ---- Train/test split one more time for confusion matrix ----
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=random_state
    )

    best_depth, best_min_samples = best_params
    final_clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=best_depth,
        min_samples_leaf=best_min_samples,
        random_state=random_state
    )
    final_clf.fit(X_train, y_train, sample_weight=w_train)

    # ---- Confusion matrix ----
    y_pred = final_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=final_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_clf.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix for Best Decision Tree")
    plt.show()

    # ---- Visualise tree ----
    plt.figure(figsize=(16, 10))
    plot_tree(
        final_clf,
        feature_names=selected_features,
        class_names=final_clf.classes_,
        filled=True,
        fontsize=6
    )
    plt.show()

    return final_clf, mean_acc


if __name__ == "__main__":
    selected_features = select_features(quick_data().drop(columns=['persid']))
    print("Selected features:", selected_features)
    run_decision_tree(selected_features, n_boot=50, n_jobs=-1)
