print("Running LMI.py")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing import quick_data
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.utils import resample

result = quick_data()

#Resample data to ensure balanced classes
clean_df = result[["agegroup_fine", "overall_trip_efficiency", "persinc_fine", "totalwfh_ord","most_used_mode", "travel_time"]].dropna() #FIX THIS
clean_df = pd.concat([
    resample(df, n_samples=225, random_state=42, replace=False)
    for
      _, df in clean_df.groupby("most_used_mode")
])

# Scale
scaler = MinMaxScaler()
norm_data = pd.DataFrame(
    scaler.fit_transform(clean_df.drop(columns="most_used_mode")),
    columns=clean_df.columns[:-1]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    norm_data,
    clean_df["most_used_mode"],
    test_size=0.2,
    random_state=42
)
#Creates the knn Classifier
knn = KNeighborsClassifier(n_neighbors=4)  # You can change the number of neighbors
knn.fit(X_train, y_train)

# Demonstrates accuracy of KNN model
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for KNN")
plt.show()


error_rates = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rates.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 21), error_rates, marker='o', linestyle='--', color='blue')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show() 
