# Crop Yield Prediction — Run Instructions

## What's in this folder
- `generate_dataset.py` → creates `yield_df.csv` (the data — same column
  layout as the real Kaggle "Crop Yield Prediction Dataset")
- `train_model.py` → cleans the data, builds a preprocessing pipeline
  (OneHotEncoder + StandardScaler via ColumnTransformer), trains 4 models
  (Linear Regression, Lasso, Decision Tree, Random Forest), and saves the
  best one as `crop_yield_model.pkl`
- `app.py` → the Streamlit web app that uses the saved model

## First-time setup (run once)
Open a terminal in this folder and run:

```
pip install pandas numpy scikit-learn joblib streamlit
```

## Every time you want to run the project
Run these THREE commands, in this exact order:

```
python generate_dataset.py
python train_model.py
streamlit run app.py
```

The third command opens the app in your browser automatically
(usually at http://localhost:8501).

## What to screenshot for your report
1. The terminal output of `train_model.py` (the model comparison table)
   → use in Chapter 8 "Test Cases and Test Results"
2. The Streamlit app itself with a filled-in form and a prediction showing
   → use in Chapter 6 "Screenshots"
3. Try 3-4 different inputs (different crop/season/rainfall) and
   screenshot each result → these are your "test cases"

## If something breaks
- "ModuleNotFoundError" → you skipped the pip install step above
- App loads but shows an error about `crop_yield_model.pkl` not found
  → you ran `streamlit run app.py` before running the first two scripts
- Nothing else should go wrong — the dataset is generated locally,
  no internet or API keys needed.

## For your report's "Data Source" note
This project uses a synthetically generated dataset (via
`generate_dataset.py`) shaped exactly like the real Kaggle
"Crop Yield Prediction Dataset" — same column names: Area (country),
Item (crop), Year, average_rain_fall_mm_per_year, pesticides_tonnes,
avg_temp, and hg/ha_yield (target). This was done so the project runs
with zero external downloads or setup. If your teacher wants the real
file, search Kaggle for "Crop Yield Prediction Dataset" and drop the
downloaded CSV in as `yield_df.csv` — no other code needs to change
since the column names already match.

## Note on hg/ha (the yield unit)
The real dataset measures yield in hectograms per hectare (hg/ha) —
that's just the unit the original data uses, not a mistake. 1 hg/ha
= 0.0001 tons/hectare, so 30,000 hg/ha = 3 tons/hectare. The app shows
both numbers so it's easy to sanity-check.
