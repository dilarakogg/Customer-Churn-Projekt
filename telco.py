import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. DATA LOADING CLEANING
# ==========================================
print("--- Loading and Cleaning Data ---")
df = pd.read_csv("telco_customer.csv", sep=';')

# missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)


print("--- Starting Visual Data Analysis ---")

# Plot 1: Overall Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title("General Customer Churn Distribution")
plt.show() 

# Plot 2: Churn by Contract Type 
plt.figure(figsize=(10, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='magma')
plt.title("Churn Rate Analysis by Contract Type")
plt.show()

# Plot 3: Monthly Charges Density
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True)
plt.title("Impact of Monthly Charges on Customer Churn")
plt.show()

print("--- EDA Completed. Transitioning to Modeling Phase ---")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# Adding custom features to improve model intelligence
df.drop("customerID", axis=1, inplace=True)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTerm"] = np.where(df["tenure"] > 24, 1, 0)

services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["TotalServices"] = df[services].apply(lambda x: sum(x != "No"), axis=1)

df["HasAutoPayment"] = np.where(df["PaymentMethod"].str.contains("automatic"), 1, 0)
df["ContractType"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

# Encoding the target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ==========================================
# 4. MACHINE LEARNING Pipeline
# ==========================================
y = df["Churn"]
X = df.drop("Churn", axis=1)

# Splitting the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Define column types for preprocessing
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

#  Scaling and Encoding
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Build Pipeline - Random Forest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# ==========================================
# 5. MODEL TRAINING & OPTIMIZATION
# ==========================================
print("Training Model with GridSearchCV... Please wait.")
param_grid = {
    'classifier__max_depth': [5, 10],
    'classifier__n_estimators': [100, 200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# ==========================================
# EVALUATION 
# ==========================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n--- FINAL MODEL PERFORMANCE REPORT ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Metrics:")
print(classification_report(y_test, y_pred))

#  (Proving accuracy)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues', ax=ax)
plt.title("Model Prediction: Success vs Error Rate")
plt.show()

# Plot 5: Feature Importance 
cat_names = best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_names)
importances = pd.Series(best_model.named_steps['classifier'].feature_importances_, index=all_feature_names)

plt.figure(figsize=(10, 8))
importances.sort_values(ascending=False).head(10).plot(kind="barh", color='skyblue')
plt.title("Top 10 Most Important Factors for Churn Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n Analysis and Modeling Completed Successfully!")