import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing import quick_data
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


result = quick_data()

# Specify data to be used in analysis
KNN_df = result[["agegroup_fine", "overall_trip_efficiency", "wasted_time", "persinc_fine", "totalwfh_ord", "travel_time"]]

# Scales required Data to prevent bias
scaler = MinMaxScaler()
norm_KNN_data = pd.DataFrame(scaler.fit_transform(KNN_df.dropna()), columns=KNN_df.columns)

# Splits data into train and test
X_train, X_test, y_train, y_test = train_test_split(norm_KNN_data.dropna(), result.dropna()["most_used_mode"], test_size=0.2, random_state=42)

#Creates the knn Classifier
knn = KNeighborsClassifier(n_neighbors=6)  # You can change the number of neighbors
knn.fit(X_train, y_train)

# Demonstrates accuracy of KNN model with classification report parameters
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

#Creates the graph to find the optimal K value
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
