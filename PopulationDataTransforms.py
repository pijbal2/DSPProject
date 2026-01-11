import pandas as pd

# ---- 1. File paths (Linux Downloads folder) ----
input_file = "/home/pejuz/Downloads/liepos1gyv.xlsx"
output_file = "/home/pejuz/Downloads/municipalities_population_long.xlsx"

# ---- 2. Load file ----
df = pd.read_excel(input_file)

# ---- 3. Extract year information ----
year_columns = df.columns[2:6]
years = df.iloc[2, 2:6].astype(int).tolist()
year_map = dict(zip(year_columns, years))

# ---- 4. Prepare data ----
data = df.iloc[4:].copy()
data = data.rename(columns={data.columns[0]: "municipality"})
data = data[["municipality"] + list(year_columns)]

# Remove "Miestas ir kaimas"
data = data[data[year_columns[0]] != "Miestas ir kaimas"]

# Keep only municipalities
data = data[data["municipality"].str.endswith("sav.", na=False)]

# ---- 5. Reshape to long format ----
long = data.melt(
    id_vars="municipality",
    value_vars=year_columns,
    var_name="year_col",
    value_name="population"
)

# ---- 6. Create date column ----
long["year"] = long["year_col"].map(year_map)
long["date"] = pd.to_datetime(long["year"].astype(str) + "-07-01")

final_df = (
    long[["municipality", "date", "population"]]
    .sort_values(["municipality", "date"])
    .reset_index(drop=True)
)

print(final_df)
# ---- 7. Save result to Downloads ----
final_df.to_excel(output_file, index=False)

print("Saved to:", output_file)
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---- 1. File paths ----
input_file = "/home/pejuz/Downloads/metupradziagyv.xlsx"
output_file = "/home/pejuz/Downloads/municipalities_population_long_yearstart.xlsx"

# ---- 2. Load file ----
df = pd.read_excel(input_file)

# ---- 3. Extract year info (same structure as previous file) ----
year_columns = df.columns[2:6]
years = df.iloc[2, 2:6].astype(int).tolist()
year_map = dict(zip(year_columns, years))

# ---- 4. Prepare data ----
data = df.iloc[4:].copy()
data = data.rename(columns={data.columns[0]: "municipality"})
data = data[["municipality"] + list(year_columns)]

# Remove "Miestas ir kaimas"
data = data[data[year_columns[0]] != "Miestas ir kaimas"]

# Keep only municipalities
data = data[data["municipality"].str.endswith("sav.", na=False)]

# ---- 5. Convert to long format ----
long = data.melt(
    id_vars="municipality",
    value_vars=year_columns,
    var_name="year_col",
    value_name="population"
)

# ---- 6. Create date column (January 1st) ----
long["year"] = long["year_col"].map(year_map)
long["date"] = pd.to_datetime(long["year"].astype(str) + "-01-01")

final_df = (
    long[["municipality", "date", "population"]]
    .sort_values(["municipality", "date"])
    .reset_index(drop=True)
)
print(final_df)
# ---- 7. Save ----
final_df.to_excel(output_file, index=False)

print("Saved to:", output_file)
