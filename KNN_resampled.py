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

EDUCATION_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
WORK_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
STOPS_COL = ["persid","travtime", "vistadist" , "duration"]



# Load the dataset
work_trip = pd.read_csv("datasets/journey_education.csv", usecols=WORK_COL)
education_trip = pd.read_csv("datasets/journey_work.csv", usecols=EDUCATION_COL)
stops = pd.read_csv("datasets/stops.csv", usecols=STOPS_COL)

# initialises data with preprocessing functions
result = quick_data()



# Create binning categories for modes of transport
mapping = {
    "Bicycle": "Active", "Mobility Scooter": "Active", "Motorcycle": "Private",
    "Public Bus": "Public", "Rideshare Service": "Public", "School Bus": "Public",
    "Taxi": "Private", "Train": "Public", "Tram": "Public",
    "Vehicle Driver": "Private", "Vehicle Passenger": "Private", "Walking": "Active", "Other": "Private",
    "Plane" : "Public", "Running/jogging" : "Active"
}

result["most_used_mode"] = result["most_used_mode"].replace(mapping)

#Resample data to ensure balanced classes
clean_df = result[["agegroup", "overall_trip_efficiency", "persinc", "totalwfh","most_used_mode"]].dropna()
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
