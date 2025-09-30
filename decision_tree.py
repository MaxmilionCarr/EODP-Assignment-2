# tree_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from preprocessing import quick_data
from MI import select_features

def run_decision_tree(selected_features, target_col='most_used_mode', random_state=42):
    # Load and split data
    df = quick_data().drop(columns=['persid'])
    weightings = df['perspoststratweight'] if 'perspoststratweight' in df.columns else None
    X = df[selected_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weightings, test_size=0.33, stratify=y, random_state=random_state
    )

    # Train decision tree using entropy (information gain)
    best = (None, 0)  # (best_tree, best_accuracy)
    for depth in range(1, 10):
        for min_samples in range(1, 6):
            print(f"\nTraining Decision Tree with max_depth={depth}")
            clf = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,            # adjust to avoid overfitting
                min_samples_leaf=min_samples,        # can help generalisation
                random_state=random_state
            )
            clf.fit(X_train, y_train, sample_weight=w_train)
            
            # Evaluate on test data
            y_pred = clf.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            print(f"Test Accuracy: {accuracy:.4f} at depth {depth}")
            if accuracy > best[1]:
                best = (clf, accuracy)
                print(f"New best model found with accuracy {accuracy:.4f}")
    
    clf = best[0]
    print("\nBest Decision Tree Model Report:")
    print(classification_report(y_test, best[0].predict(X_test)))

    # Visualise the tree
    plt.figure(figsize=(16, 10))
    plot_tree(
        clf,
        feature_names=selected_features,
        class_names=clf.classes_,
        filled=True,
        fontsize=5
    )
    plt.show()

    return clf

if __name__ == "__main__":
    selected_features = select_features(quick_data().drop(columns=['persid']))
    print("Selected features:", selected_features)
    run_decision_tree(selected_features)
