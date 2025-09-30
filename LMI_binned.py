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
EDUCATION_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
WORK_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
STOPS_COL = ["persid","travtime", "vistadist" , "duration"]



# Load the dataset
work_trip = pd.read_csv("datasets/journey_education.csv", usecols=WORK_COL)
education_trip = pd.read_csv("datasets/journey_work.csv", usecols=EDUCATION_COL)
stops = pd.read_csv("datasets/stops.csv", usecols=STOPS_COL)

# initialises data with preprocessing functions
result = quick_data()

#Compute travel efficiency as the average of (distance / travel time) across all trips for each individual (persid).

work_trip["wasted_time_work"] = pd.to_numeric(work_trip["journey_elapsed_time"], errors='coerce') - pd.to_numeric(work_trip["journey_travel_time"], errors='coerce')
work_trip["distance_work"] = work_trip["journey_distance"]
work_trip["time_work"] = work_trip["journey_travel_time"]
overall_work_time_waste = work_trip[["persid", "wasted_time_work", "distance_work","time_work"]].drop_duplicates()

education_trip["wasted_time_education"] = pd.to_numeric(education_trip["journey_elapsed_time"], errors='coerce') - pd.to_numeric(education_trip["journey_travel_time"], errors='coerce')
education_trip["distance_education"] = education_trip["journey_distance"]
education_trip["time_education"] = education_trip["journey_travel_time"]
overall_education_time_waste = education_trip[["persid", "wasted_time_education", "distance_education","time_education"]].drop_duplicates()

merged = pd.merge(overall_education_time_waste, overall_work_time_waste, on="persid", how="outer")

# Create binning categories for modes of transport
mapping = {
    "Bicycle": "Active", "Mobility Scooter": "Active", "Motorcycle": "Private",
    "Public Bus": "Public", "Rideshare Service": "Public", "School Bus": "Public",
    "Taxi": "Private", "Train": "Public", "Tram": "Public",
    "Vehicle Driver": "Private", "Vehicle Passenger": "Private", "Walking": "Active", "Other": "Private",
    "Plane" : "Public", "Running/jogging" : "Active"
}

result["most_used_mode"] = result["most_used_mode"].replace(mapping)
# Pick work if it exists, otherwise education
merged["wasted_time"] = merged["wasted_time_work"].combine_first(merged["wasted_time_education"])
merged["distance"] = merged["distance_work"].combine_first(merged["distance_education"])
merged["time"] = merged["time_work"].combine_first(merged["time_education"])
# Final tidy dataframe
overall_time_waste = merged[["persid", "wasted_time", "distance","time"]]

print("Shapes before merge:", result.shape, overall_education_time_waste.shape, overall_work_time_waste.shape, overall_time_waste.shape)

result = result.merge(overall_time_waste, on='persid', how='left')


print(result.columns)
types_df = result.iloc[0].map(type)
for col in result.columns:
    print("Type of", col, ":", types_df[col])
    print("Example data in", col, ":", result[col].dropna().values[:5])

print("Data shape:", result.shape)
for col in result.columns:
    print("Nan in", col, ":", result[col].isna().sum())

KNN_df = result[["agegroup", "overall_trip_efficiency", "wasted_time", "distance", "persinc", "totalwfh","time"]]

# Scales required Data to prevent bias
scaler = MinMaxScaler()
norm_KNN_data = pd.DataFrame(scaler.fit_transform(KNN_df.dropna()), columns=KNN_df.columns)

# Splits data into train and test
X_train, X_test, y_train, y_test = train_test_split(norm_KNN_data.dropna(), result.dropna()["most_used_mode"], test_size=0.2, random_state=42)

#Creates the knn Classifier
knn = KNeighborsClassifier(n_neighbors=6)  # You can change the number of neighbors
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