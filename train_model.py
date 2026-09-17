"""
train_model.py
---------------
Loads yield_df.csv, cleans it, builds a preprocessing + model PIPELINE
(same technique shown in the video), trains 3 models, compares them,
and saves the best one.

Run this SECOND, after generate_dataset.py.

WHY A "PIPELINE" WITH A "COLUMNTRANSFORMER"? (explain this in your report)
Our data has two kinds of columns:
  - text columns (Area, Item)       -> need OneHotEncoding (turns each
                                        country/crop into 0/1 columns,
                                        since ML models can't read text)
  - number columns (Year, rainfall,
    pesticides, avg_temp)           -> need Scaling (so no single big
                                        number like "rainfall=3000"
                                        unfairly dominates over a small
                                        one like "temp=25")
A ColumnTransformer applies the RIGHT transformation to the RIGHT
columns, all in one step. A Pipeline then chains that transformer +
the model together, so later we can call one single .predict() and
it handles both steps automatically -- that's exactly what the app
needs to do.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---- 1. Load data ----
df = pd.read_csv("yield_df.csv")

# ---- 2. Basic cleaning (same steps shown in the video) ----
print(f"Rows before cleaning: {len(df)}")
df = df.drop_duplicates()          # remove exact duplicate rows
df = df.dropna()                   # remove rows with missing values
print(f"Rows after cleaning:  {len(df)}")

# ---- 3. Split features (X) and target (y) ----
categorical_cols = ["Area", "Item"]
numeric_cols = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

X = df[categorical_cols + numeric_cols]
y = df["hg/ha_yield"]

# ---- 4. Build the ColumnTransformer ----
# categorical columns -> OneHotEncoder, numeric columns -> StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# ---- 5. Train/test split (80% train, 20% test) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 6. Train 3 models, each wrapped in the SAME preprocessing pipeline ----
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results[name] = (r2, mae)
    fitted_pipelines[name] = pipeline

# ---- 7. Print comparison table -- SCREENSHOT THIS for your report! ----
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':<20}{'R2 Score':<15}{'MAE (error)'}")
for name, (r2, mae) in results.items():
    print(f"{name:<20}{r2:<15.3f}{mae:.1f}")

# ---- 8. Pick the best model (highest R2) and save its whole pipeline ----
best_name = max(results, key=lambda n: results[n][0])
best_pipeline = fitted_pipelines[best_name]

print(f"\nBest model: {best_name} -- saved to crop_yield_model.pkl")

joblib.dump(
    {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "area_options": sorted(df["Area"].unique()),
        "item_options": sorted(df["Item"].unique()),
    },
    "crop_yield_model.pkl",
)
