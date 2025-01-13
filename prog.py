import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the data
dataLocation = ""  # input location of data file here
df = pd.read_excel(dataLocation)
df.fillna(0, inplace=True)

# Extract relevant data
start, end = 2, 531
cols = ['Location', 'pH', 'EC', 'TDS', 'ORP', 'Ca', 'Mg', 'Na', 'K', 'HCO3', 'SO4', 'Cl', 'NO3', 'CO3', 'F', 'Cu', 'Pb', 'Zn', 'U', 'NH4', 'Br', 'Date']
df = df.iloc[start:end, :].reset_index(drop=True)
df.columns = cols

# Define a helper function to calculate values
def apply_normalization(series, ranges):
    """
    Normalize a series based on predefined ranges.
    """
    for value, range_tuple in ranges:
        if range_tuple[0] <= series <= range_tuple[1]:
            return value
    return 0

# Define ranges for all columns
normalization_ranges = {
    'pH': [(100, (7, 8.5)), (80, (6.8, 8.5)), (60, (6.7, 8.8)), (40, (6.5, 9))],
    'EC': [(100, (1800, 1900)), (80, (1500, 2200)), (60, (1200, 2500)), (40, (900, 2800))],
    'TDS': [(100, (500, 1500)), (80, (400, 2000)), (60, (300, 3000)), (40, (200, 3500))],
    # You can add other columns similarly...
}

# Apply normalization using helper function
for col, ranges in normalization_ranges.items():
    df[f'n{col}'] = df[col].apply(lambda x: apply_normalization(x, ranges))

# Weighted normalization based on column mapping
weights = {
    'pH': 0.08, 'EC': 0.004, 'TDS': 0.053, 'ORP': 0.053, 'Ca': 0.053, 'Mg': 0.053, 'Na': 0.014, 
    'K': 0.053, 'HCO3': 0.053, 'SO4': 0.053, 'Cl': 0.053, 'NO3': 0.053, 'CO3': 0.053, 'F': 0.053, 
    'Cu': 0.053, 'Pb': 0.053, 'Zn': 0.053, 'U': 0.053, 'NH4': 0.053, 'Br': 0.053
}

# Apply weighted normalization
for col, weight in weights.items():
    df[f'w{col}'] = df[f'n{col}'] * weight

# Calculate the WQI
df['wqi'] = df[[f'w{col}' for col in weights]].sum(axis=1)

# Convert Date to int64 for ML model
df['Date'] = df['Date'].values.astype('int64')

# Aggregation
aggr = df.groupby('Date')['wqi'].mean().reset_index()
aggr = aggr[np.isfinite(aggr['wqi'])]

# Train-test split
X = aggr[['Date']]
y = aggr['wqi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Linear Regression Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Evaluate Model
init_mse = mean_squared_error(y_test, y_pred)
acc = r2_score(y_test, y_pred) * 100
final_mse = np.sqrt(init_mse)

# Display stats
print(f"\n\nStats :-")
print(f"1) Mean squared error : {init_mse:.2f}")
print(f"2) Accuracy : {acc:.2f} %")

# Plotting
dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dt = pd.concat([X_test, dt], axis=1)
dt['Date'] = pd.to_datetime(dt['Date'])

plt.scatter(dt['Date'], dt['Actual'])
plt.plot(dt['Date'], dt['Predicted'], color='b')
plt.title("Linear Regression")
plt.xlabel("Date")
plt.ylabel("WQI")
plt.show()
