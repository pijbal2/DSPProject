import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Loading of dataset
downloads_path = os.path.expanduser("~/Downloads")
df = pd.read_excel(os.path.join(downloads_path, "dogcattimeseries2023_ratio.xlsx"))

# Let's check the correlation matrix of all our numeric variables to see which variables we can discard:
numeric_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix - All Numeric Variables', fontsize=16)
plt.show()




# Let's only keep relevant features for the household pet dataset:
selected_cols = [
    "house_apartmentratio",
    "Turto objektų skaičius_vienodviejubutu",
    "Turto objektų skaičius_trijuirdaugiaubutu",
    "total_flats_2024_04",
    "people_per_flat",
    "cat_dog_ratio"
]
df_model = df[selected_cols].copy()
df_model.rename(columns={
    "Turto objektų skaičius_vienodviejubutu": "vienodviejubutu_count",
    "Turto objektų skaičius_trijuirdaugiaubutu": "trijuirdaugiaubutu_count"
}, inplace=True)


# Correlation matrix for selected features
plt.figure(figsize=(10, 8))
sns.heatmap(df_model.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix - Selected Features', fontsize=16)
plt.show()


# Let's define the features and target, do a train test split and train a random forest model and see its performance:
X = df_model[[
    "house_apartmentratio",
    "vienodviejubutu_count",
    "trijuirdaugiaubutu_count",
    "total_flats_2024_04",
    "people_per_flat"
]]
y = df_model["cat_dog_ratio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Let's train a simple linear regression model and a random forest regressor and compare results:
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print("R2 score:", round(r2_lr, 3))
print("RMSE:", round(rmse_lr, 3))

simple_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
simple_rf_model.fit(X_train, y_train)
y_pred_rf_simple = simple_rf_model.predict(X_test)

mse_rf_simple = mean_squared_error(y_test, y_pred_rf_simple)
rmse_rf_simple = np.sqrt(mse_rf_simple)
r2_rf_simple = r2_score(y_test, y_pred_rf_simple)

print("\nRandom Forest Results (100 trees):")
print("R2 score:", round(r2_rf_simple, 3))
print("RMSE:", round(rmse_rf_simple, 3))


# Grid search for Random Forest n_estimators from 100 to 2000 in steps of 10
grid_results = []

for n in range(100, 2001, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    grid_results.append({"n_estimators": n, "R2": r2, "RMSE": rmse})

# Let's perform a grid search for our random forest model and visualize the results:
grid_df = pd.DataFrame(grid_results)
print("\nGrid Search Results:")
print(grid_df)

best_row = grid_df.loc[grid_df["R2"].idxmax()]
print("\nBest Random Forest from Grid Search:")
print(best_row)


fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('R2', color=color)
ax1.plot(grid_df['n_estimators'], grid_df['R2'], marker='o', color=color, label='R2')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(grid_df['n_estimators'][::10]) 

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('RMSE', color=color)
ax2.plot(grid_df['n_estimators'], grid_df['RMSE'], marker='x', linestyle='--', color=color, label='RMSE')
ax2.tick_params(axis='y', labelcolor=color)

ax1.axvline(best_row['n_estimators'], color='green', linestyle=':', label='Best n_estimators')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig.suptitle('Random Forest Performance vs n_estimators', fontsize=14)
fig.tight_layout()

plt.show() 


# Random Forest feature importance for n_estimators = 430
rf_430 = RandomForestRegressor(n_estimators=430, random_state=42)
rf_430.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_430.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance (Random Forest, 430 trees):")
print(feature_importance)



# Let's plot a decision tree to gain a better understanding of how the model makes choices:
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42) 
dt_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=12)
plt.title('Decision Tree Approximation of Random Forest', fontsize=16)
plt.show()
