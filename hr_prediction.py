from sqlalchemy import create_engine

username = "root"
password = "deepa1810"
host = "localhost"
database = "hr_prediction_db"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)

with engine.connect() as connection:
    print("MySQL connection successful!")

    import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("HR_Prediction_Cleaned.csv")

print("Dataset shape:", df.shape)

# Insert data into MySQL
#df.to_sql(
    #name="candidate_records",
#    con=engine,
#    if_exists="replace",
 #   index=False,
 #   chunksize=1000
#)

#print("Data inserted successfully!")

with engine.connect() as connection:
    result = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM candidate_records"
    )

    total = result.scalar()
print("Total records in MySQL:", total)

#import pandas as pd

# Load the cleaned dataset
#df = pd.read_csv("HR_Prediction_Cleaned.csv")

# Convert target labels into numbers
#df["placed_flag"] = df["status"].map({
#    "Placed": 1,
#    "Not Placed": 0
#})

#print("Target distribution:")
#print(df["placed_flag"].value_counts())

#print("\nMissing target values:", df["placed_flag"].isna().sum())

#from sklearn.model_selection import train_test_split

# Define target
#y = df["placed_flag"]

# Define input features
#X = df.drop(columns=["status", "placed_flag"])

# Remove ID columns if present
#X = X.drop(
#   columns=["candidate_id", "id"],
 
#   errors="ignore"
#)

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(
#    X,
#    y,
#    test_size=0.2,
#    random_state=42,
#    stratify=y
#)

#print("Training records:", X_train.shape[0])
#print("Testing records:", X_test.shape[0])
#print("Training features:", X_train.shape[1])
from sklearn.model_selection import train_test_split
# Create target variable
df["placed_flag"] = df["status"].map({
    "Placed": 1,
    "Not Placed": 0
})

print("\nTarget distribution:")
print(df["placed_flag"].value_counts())

print("Missing target values:", df["placed_flag"].isna().sum())
# Target variable
y = df["placed_flag"]

# Input features
X = df.drop(columns=["status", "placed_flag"])

# Remove identifier columns if present
X = X.drop(
    columns=["candidate_id", "id"],
    errors="ignore"
)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Data Split Results ---")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Training features:", X_train.shape[1])

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess numerical data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical data
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib
import pandas as pd

# Save the trained model
joblib.dump(model, "hr_prediction_model.pkl")

# Create prediction results
results = pd.DataFrame({
    "Actual_Status": y_test.map({
        0: "Not Placed",
        1: "Placed"
    }).values,
    "Predicted_Status": pd.Series(y_pred, index=y_test.index).map({
        0: "Not Placed",
        1: "Placed"
    }).values,
    "Placement_Probability_Percent": (
        y_prob * 100
    ).round(2)
})

# Save results to CSV
results.to_csv("HR_Prediction_Results.csv", index=False)

print("\nModel saved: hr_prediction_model.pkl")
print("Prediction results saved: HR_Prediction_Results.csv")
print("Prediction results shape:", results.shape)
