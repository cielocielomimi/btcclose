import os, json, math, time, datetime as dt
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_poisson_deviance, mean_gamma_deviance

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------- Settings --------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1d")
CSV_PATH = os.getenv("CSV_PATH", "data/oct15.csv")
TIME_STEP = int(os.getenv("TIME_STEP", "15"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "1e-3"))
DROPOUT = float(os.getenv("DROPOUT", "0.0"))
ACT = os.getenv("ACT", "tanh")

# -------- Utilities --------
def fetch_latest_binance_data(symbol="BTCUSDT", interval="1d", start_date=None, limit=1000):
    """Public klines (no API key)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_date is not None:
        params["startTime"] = int(pd.Timestamp(start_date).timestamp() * 1000)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(data, columns=[
        "Open Time","Open","High","Low","Close","Volume","Close Time",
        "Quote Volume","Number of Trades","Taker Buy Base","Taker Buy Quote","Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
    df = df[["Date","Open","High","Low","Close","Volume"]].astype({"Open":float,"High":float,"Low":float,"Close":float,"Volume":float})
    return df

def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def evaluation(model, x_train, y_train, x_test, y_test, scaler):
    # preds (scaled)
    train_pred = model.predict(x_train, verbose=0)
    test_pred  = model.predict(x_test, verbose=0)

    # inverse to prices
    train_pred_i = scaler.inverse_transform(train_pred)
    test_pred_i  = scaler.inverse_transform(test_pred)
    ytrain_i     = scaler.inverse_transform(y_train.reshape(-1,1))
    ytest_i      = scaler.inverse_transform(y_test.reshape(-1,1))

    eps = 1e-8
    ytest_pos      = np.clip(ytest_i, eps, None)
    test_pred_pos  = np.clip(test_pred_i, eps, None)
    ytrain_pos     = np.clip(ytrain_i, eps, None)
    train_pred_pos = np.clip(train_pred_i, eps, None)

    def mape(y, yhat):
        mask = np.abs(y) > eps
        return float(np.mean(np.abs((y[mask]-yhat[mask])/y[mask]))) if mask.any() else float("nan")

    def smape(y, yhat):
        return float(2*np.mean(np.abs(yhat-y) / (np.abs(y)+np.abs(yhat)+eps)))

    metrics = {
        "Metric": ["RMSE","MSE","MAE","MAPE","sMAPE","R²","Explained Variance","Gamma Deviance","Poisson Deviance"],
        "Train": [
            float(np.sqrt(mean_squared_error(ytrain_i, train_pred_i))),
            float(mean_squared_error(ytrain_i, train_pred_i)),
            float(mean_absolute_error(ytrain_i, train_pred_i)),
            mape(ytrain_i, train_pred_i),
            smape(ytrain_i, train_pred_i),
            float(r2_score(ytrain_i, train_pred_i)),
            float(explained_variance_score(ytrain_i, train_pred_i)),
            float(mean_gamma_deviance(ytrain_pos, train_pred_pos)),
            float(mean_poisson_deviance(ytrain_pos, train_pred_pos)),
        ],
        "Test": [
            float(np.sqrt(mean_squared_error(ytest_i, test_pred_i))),
            float(mean_squared_error(ytest_i, test_pred_i)),
            float(mean_absolute_error(ytest_i, test_pred_i)),
            mape(ytest_i, test_pred_i),
            smape(ytest_i, test_pred_i),
            float(r2_score(ytest_i, test_pred_i)),
            float(explained_variance_score(ytest_i, test_pred_i)),
            float(mean_gamma_deviance(ytest_pos, test_pred_pos)),
            float(mean_poisson_deviance(ytest_pos, test_pred_pos)),
        ]
    }
    return pd.DataFrame(metrics)

# -------- Load + update data --------
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"ERROR: CSV not found at {CSV_PATH}")

btcdf = pd.read_csv(CSV_PATH)
if "Adj Close" in btcdf.columns:
    btcdf = btcdf.drop(columns=["Adj Close"])

btcdf["Date"] = pd.to_datetime(btcdf["Date"])
last_date = btcdf["Date"].max() + pd.Timedelta(days=1)

try:
    new_data = fetch_latest_binance_data(symbol=SYMBOL, interval=INTERVAL, start_date=last_date)
    if not new_data.empty:
        btcdf = pd.concat([btcdf, new_data], ignore_index=True)
        btcdf = btcdf.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)
except Exception as e:
    print("[WARN] Binance fetch failed:", e)

btcdf["Date"] = pd.to_datetime(btcdf["Date"]).dt.date
closedf = pd.DataFrame({"Date": pd.to_datetime(btcdf["Date"]), "Close": btcdf["Close"].astype(float)})
closedf_copy = closedf.copy()

# -------- Scale + split --------
scaler = MinMaxScaler(feature_range=(0,1))
series = scaler.fit_transform(closedf[["Close"]].values)
n = len(series)
train_end = int(n*0.80)
train_data, test_data = series[:train_end], series[train_end:]

x_train, y_train = create_dataset(train_data, TIME_STEP)
x_test,  y_test  = create_dataset(test_data,  TIME_STEP)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0],  x_test.shape[1],  1)

# -------- Model --------
neurons = x_train.shape[1]*8
model = Sequential([
    Input(shape=(x_train.shape[1], x_train.shape[2])),
    LSTM(neurons, activation=ACT),
    Dropout(DROPOUT),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=LR), loss="mse",
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                       tf.keras.metrics.MeanAbsolutePercentageError(name="mape")])
early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early], verbose=0)

# -------- Evaluation --------
results_df = evaluation(model, x_train, y_train, x_test, y_test, scaler)

# -------- Next-day forecast (one step) --------
x_last = series[-TIME_STEP:].reshape(1, TIME_STEP, 1)
next_day_scaled = model.predict(x_last, verbose=0)
next_day_close_forecast = float(scaler.inverse_transform(next_day_scaled)[0,0])

# -------- Outputs --------
run_ts_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat()
last_obs_date = closedf_copy["Date"].max()
target_close_date_utc = (last_obs_date + pd.Timedelta(days=1)).date().isoformat()

metrics_test = results_df.set_index("Metric")["Test"].astype(float).to_dict()

payload = {
    "symbol": SYMBOL,
    "horizon": "next_daily_close",
    "run_ts_utc": run_ts_utc,
    "target_close_date_utc": target_close_date_utc,
    "forecast_close": next_day_close_forecast,
    "metrics_h1": metrics_test
}

os.makedirs("out", exist_ok=True)
with open("out/daily_forecast.json", "w") as f:
    json.dump(payload, f, indent=2)

row = {"run_ts_utc": run_ts_utc, "target_close_date_utc": target_close_date_utc,
       "symbol": SYMBOL, "forecast_close": next_day_close_forecast,
       **{f"h1_{k.lower().replace(' ','_')}": float(v) for k,v in metrics_test.items()}}
hist_path = "out/history.csv"
if os.path.exists(hist_path):
    dfh = pd.read_csv(hist_path)
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
else:
    dfh = pd.DataFrame([row])
dfh.to_csv(hist_path, index=False)

# logs for Actions
print(json.dumps(payload, indent=2))
print("[OK] Wrote out/daily_forecast.json and out/history.csv")
print(f"Next-day forecast close: {next_day_close_forecast:.2f}")
