import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from pmdarima import auto_arima

# Load data
contract_file_path = "all_new_variables.xlsx"

df = pd.read_excel(contract_file_path)

# Filter data for 2014-2019
filtered_df = df[(df['shipping_date'] >= '2014-01-01') & (df['shipping_date'] <= '2019-12-31')].copy()

# Prepare data for regression models
X_columns = [
    'distance', 'number_of_bushels',
    'index_difference', 'contract_to_shipping', 'shipping_to_due',
    'future_contract_month_adjusted_DEC',
    'future_contract_month_adjusted_JUL',
    'future_contract_month_adjusted_MAR',
    'future_contract_month_adjusted_MAY',
    'future_contract_month_adjusted_SEP', 'recency', 'tenure', 'frequency',
    'basis_to_dec_wnye', 'duration_DEC', 'duration_MAR', 'duration_MAY',
    'duration_JUL', 'duration_SEP'
]

X = filtered_df[X_columns]
y = filtered_df['basis_to_dec']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(random_state=1).fit(X_train, y_train)
xgb_model = XGBRegressor(random_state=1).fit(X_train, y_train)
adaboost_model = AdaBoostRegressor(random_state=1, n_estimators=100).fit(X_train, y_train)

# Time series data preparation for SARIMA
df['shipping_date'] = pd.to_datetime(df['shipping_date'])
df.set_index('shipping_date', inplace=True)
df_monthly = df['basis_to_dec'].resample('M').mean()

# Train SARIMA model
sarima_train = df_monthly.iloc[:-12]
sarima_model = auto_arima(sarima_train, seasonal=True, m=12, stepwise=True)

# Prediction function
def get_prediction(date, model_name):
    """Predicts the value using the selected model."""
    if model_name == 'LinearRegression':
        return linear_model.predict([X_test.iloc[0]])[0]
    elif model_name == 'RandomForest':
        return rf_model.predict([X_test.iloc[0]])[0]
    elif model_name == 'XGBoost':
        return xgb_model.predict([X_test.iloc[0]])[0]
    elif model_name == 'AdaBoost':
        return adaboost_model.predict([X_test.iloc[0]])[0]
    elif model_name == 'SARIMA':
        forecast = sarima_model.predict(n_periods=1)
        return forecast[0]
    else:
        raise ValueError("Unsupported model selected.")
