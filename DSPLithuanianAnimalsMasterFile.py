import pandas as pd
import os
import re


# Loading of all existing household animal data files and adding a column "source" which indicates the year month of the source 

downloads_path = os.path.expanduser("~/Downloads")

file_list = [
    os.path.join(downloads_path, f)
    for f in os.listdir(downloads_path)
    if f.startswith("laikomi_augintiniai")
]

def extract_marker(filename):
    m = re.search(r"laikomi_augintiniai_([0-9]{4}-[0-9]{2})", filename)
    return m.group(1) if m else None

dfs = []
for file in file_list:
    df = pd.read_csv(file, sep=";")
    df["source"] = extract_marker(os.path.basename(file))
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.columns = df_all.columns.str.strip()
df_all = df_all[df_all["source"].notna()]


# Aggregating the count of animals by municipality, date and source

monthly_mun_df = (
    df_all
    .groupby(
        [
            "source",
            "Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
            "Gyvūno augintinio rūšies pavadinimas"
        ],
        as_index=False
    )["Gyvūnų augintinių skaičius"]
    .sum()
)

# Defining starting period and ending period
full_months = pd.period_range(
    start="2020-01",
    end="2025-09",
    freq="M"
).astype(str)

monthly_mun_df = (
    monthly_mun_df
    .set_index(
        [
            "source",
            "Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
            "Gyvūno augintinio rūšies pavadinimas"
        ]
    )
    .unstack(fill_value=0)
    .stack()
    .reset_index()
)

monthly_mun_df = monthly_mun_df[monthly_mun_df["source"].isin(full_months)]
monthly_mun_df["month"] = monthly_mun_df["source"].astype("period[M]")

monthly_mun_df = monthly_mun_df.sort_values(
    [
        "Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        "Gyvūno augintinio rūšies pavadinimas",
        "month"
    ]
)



# Interpolating July 2022-07 values by averaging June and August
june = pd.Period("2022-06", freq="M")
july = pd.Period("2022-07", freq="M")
aug  = pd.Period("2022-08", freq="M")

def interpolate_july(g):
    if (june in g["month"].values) and (aug in g["month"].values):
        june_val = g.loc[g["month"] == june, "Gyvūnų augintinių skaičius"].iloc[0]
        aug_val  = g.loc[g["month"] == aug,  "Gyvūnų augintinių skaičius"].iloc[0]
        g.loc[g["month"] == july, "Gyvūnų augintinių skaičius"] = int(
            round((june_val + aug_val) / 2)
        )
    return g

monthly_mun_df = (
    monthly_mun_df
    .groupby(
        [
            "Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
            "Gyvūno augintinio rūšies pavadinimas"
        ],
        group_keys=False
    )
    .apply(interpolate_july)
)

# Calculating differences for the time series analysis
monthly_mun_df["monthly_diff"] = (
    monthly_mun_df
    .groupby(
        [
            "Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
            "Gyvūno augintinio rūšies pavadinimas"
        ]
    )["Gyvūnų augintinių skaičius"]
    .diff()
    .fillna(0)
)


animals_2025_09 = monthly_mun_df[monthly_mun_df["source"] == "2025-09"].copy()

# Loading and joining the number of houses data
ntr32 = pd.read_excel(
    os.path.join(downloads_path, "NTR_iregistruoti_vieno_ir_dvieju_butu_gyvenamieji_namai_ntr32.xlsx")
)
ntr33 = pd.read_excel(
    os.path.join(downloads_path, "NTR_iregistruoti_triju_ir_daugiau_butu_daugiabuciai_gyvenamieji_namai_ntr33.xlsx")
)

for df in (ntr32, ntr33):
    df.columns = df.columns.str.strip()
    df["Turto objekto savivaldybė"] = df["Turto objekto savivaldybė"].ffill()


sum_cols = [
    "Turto objektų skaičius",
    "Turto objekto bendras plotas, m²",
    "Turto objekto bendras tūris, m³",
    "Turto objekto užstatytas plotas, m²",
]

def agg_ntr(df, suffix):
    agg = df.groupby("Turto objekto savivaldybė", as_index=False)[sum_cols].sum()
    return agg.rename(columns={c: f"{c}_{suffix}" for c in sum_cols})

ntr32_agg = agg_ntr(ntr32, "vienodviejubutu")
ntr33_agg = agg_ntr(ntr33, "trijuirdaugiaubutu")

ntr_total = pd.merge(
    ntr32_agg, ntr33_agg,
    on="Turto objekto savivaldybė",
    how="outer"
)


# Getting the population data at date 2025-07 (close enough to population at date 2025-10-01)
population = pd.read_excel(os.path.join(downloads_path, "municipalities_population_long.xlsx"))
population.columns = population.columns.str.strip()

population_2025_07 = population[population["date"] == pd.Timestamp("2025-07-01")].copy()

# Getting number of flats for recent date which is 2024-04-01, also close enough for our analysis
ntr6 = pd.read_excel(
    os.path.join(downloads_path, "NTR_iregistruoti_butai_ntr6.xlsx"),
    header=2
)
ntr6.columns = ntr6.columns.astype(str).str.strip()
municipality_col = ntr6.columns[1]
date_col = next(
    col for col in ntr6.columns
    if "2024-04-01" in col
)
ntr6_2024_04 = (
    ntr6[[municipality_col, date_col]]
    .rename(columns={
        municipality_col: "municipality",
        date_col: "total_flats_2024_04"
    })
)
# -----------------------------
# 9. Join everything
# -----------------------------
final_2025_09 = (
    animals_2025_09
    .merge(
        ntr_total,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="Turto objekto savivaldybė",
        how="left"
    )
    .merge(
        population_2025_07,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="municipality",
        how="left"
    )
    .merge(
        ntr6_2024_04,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="municipality",
        how="left"
    )
)

# Transformations to properly group animals by municipality for every animal 
animal_pivot = (
    animals_2025_09
    .pivot_table(
        index="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        columns="Gyvūno augintinio rūšies pavadinimas",
        values="Gyvūnų augintinių skaičius",
        aggfunc="sum",
        fill_value=0
    )
    .reset_index()
)

# Cat / Dog ratio feature
animal_pivot["cat_dog_ratio"] = animal_pivot.get("Katė", 0) / animal_pivot.get("Šuo", 1)  # avoid div by 0
animal_pivot = animal_pivot[["Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas", "cat_dog_ratio"]]

# Getting the final dataframe for implementation of ML models
final_2025_09 = (
    animal_pivot
    .merge(
        ntr_total,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="Turto objekto savivaldybė",
        how="left"
    )
    .merge(
        population_2025_07,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="municipality",
        how="left"
    )
    .merge(
        ntr6_2024_04,
        left_on="Gyvūnų augintinio laikymo vietos savivaldybės pavadinimas",
        right_on="municipality",
        how="left"
    )
)

# Additional engineered features
final_2025_09["people_per_flat"] = final_2025_09["population"] / final_2025_09["total_flats_2024_04"]
final_2025_09["house_apartmentratio"] = (
    (final_2025_09["Turto objektų skaičius_trijuirdaugiaubutu"] +
     final_2025_09["Turto objektų skaičius_vienodviejubutu"]) /
    final_2025_09["total_flats_2024_04"]
)

# Exporting data for Power BI usage and prediction models:

output_path = os.path.join(downloads_path, "dogcattimeseries2023_ratio.xlsx")
final_2025_09.to_excel(output_path, index=False)
print(f"File exported to: {output_path}")
