"""
generate_dataset.py
--------------------
Creates yield_df.csv -- built to match the structure of the real
Kaggle "Crop Yield Prediction Dataset" (the same one used in most
YouTube tutorials, including Knowledge Doctor's Project 23 video):

    Area, Item, Year, average_rain_fall_mm_per_year,
    pesticides_tonnes, avg_temp, hg/ha_yield

Column meaning:
- Area                          -> the country
- Item                          -> the crop
- Year                          -> the year of harvest
- average_rain_fall_mm_per_year -> rainfall that year
- pesticides_tonnes             -> pesticide used (in tonnes)
- avg_temp                      -> average temperature (deg C)
- hg/ha_yield                   -> TARGET. Yield in hectograms/hectare
                                    (the real dataset's unit).
                                    1 hg/ha = 0.0001 tons/hectare,
                                    so 30000 hg/ha = 3 tons/hectare.
                                    (We convert this back to tons/ha
                                    in the app so it's easy to read.)

Run this FIRST (only once) to create the dataset.

Note for your report: the real Kaggle dataset can be downloaded from
"Crop Yield Prediction Dataset" on Kaggle if your teacher wants the
exact file the video uses. This script generates a synthetic version
with the SAME COLUMN NAMES so everything downstream (train_model.py,
app.py) works identically either way -- you can drop in the real CSV
later without changing any other code.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

AREAS = ["India", "United States of America", "Brazil", "China", "Australia"]
ITEMS = ["Wheat", "Rice, paddy", "Maize", "Potatoes", "Soybeans"]

# Typical base yield in hg/ha for each crop (rough real-world ballpark,
# matches the scale you'll see in the actual Kaggle dataset)
BASE_YIELD_HG_HA = {
    "Wheat": 30000,
    "Rice, paddy": 25000,
    "Maize": 28000,
    "Potatoes": 190000,   # potatoes yield far more mass per hectare
    "Soybeans": 25000,
}

N_ROWS = 1500
rows = []

for _ in range(N_ROWS):
    area = np.random.choice(AREAS)
    item = np.random.choice(ITEMS)
    year = int(np.random.randint(2000, 2021))
    rainfall = round(np.random.uniform(300, 3000), 1)     # mm/year
    pesticides = round(np.random.uniform(0.5, 500), 2)    # tonnes
    avg_temp = round(np.random.uniform(10, 35), 1)        # deg C

    base = BASE_YIELD_HG_HA[item]

    # Simple realistic-ish relationship, same idea as before:
    # more pesticide + decent rainfall helps a bit, extreme temperature
    # (far from ~24C) hurts yield, plus random noise
    yield_value = (
        base
        + pesticides * 20
        + rainfall * 3
        - abs(avg_temp - 24) * (base * 0.01)
        + np.random.normal(0, base * 0.08)
    )
    yield_value = max(yield_value, base * 0.2)
    yield_value = round(yield_value, 1)

    rows.append([area, item, year, rainfall, pesticides, avg_temp, yield_value])

df = pd.DataFrame(
    rows,
    columns=[
        "Area", "Item", "Year",
        "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp",
        "hg/ha_yield",
    ],
)

df.to_csv("yield_df.csv", index=False)
print(f"Done! Created yield_df.csv with {len(df)} rows.")
print(df.head())
