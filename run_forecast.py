#!/usr/bin/env python3
# coding: utf-8

# ---- Headless plotting for CI ----
import matplotlib
matplotlib.use("Agg")

import os, math, time, json, datetime as dt
import seaborn as sns
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
    explained_variance_score, r2_score, mean_poisson_deviance, mean_gamma_deviance)
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px

# ---------- CI defaults ----------
os.makedirs("out", exist_ok=True)
CSV_PATH = os.getenv("CSV_PATH", "data/oct15.csv")
TIME_STEP = int(os.getenv("TIME_STEP", "15"))

# ---------- Binance fetch ----------
def fetch_latest_binance_data(symbol="BTCUSDT", interval="1d", start_date=None):
    """Public OHLCV from Binance starting at start_date (UTC)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    if start_date is not None:
        params["startTime"] = int(pd.Timestamp(start_date).timestamp() * 1000)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
    return df

# ---------- Load CSV ----------
btcdf = pd.read_csv(CSV_PATH)
if "Adj Close" in btcdf.columns:
    btcdf.drop(columns=["Adj Close"], inplace=True)

print('Null Values:', btcdf.isnull().values.sum())
_ = btcdf.describe()  # keep for completeness, printed in logs

# ---------- Append latest from Binance ----------
btcdf['Date'] = pd.to_datetime(btcdf['Date'])
last_date = btcdf['Date'].max() + pd.Timedelta(days=1)

try:
    new_data = fetch_latest_binance_data(start_date=last_date)
    btcdf_updated = pd.concat([btcdf, new_data], ignore_index=True)
    btcdf_updated = btcdf_updated.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
except Exception as e:
    print("[WARN] Binance fetch failed:", e)
    btcdf_updated = btcdf.copy()

btcdf_updated['Date'] = pd.to_datetime(btcdf_updated['Date']).dt.date
btcdf_updated.to_csv("out/btc_updated.csv", index=False)  # save under out/
btcdf = btcdf_updated.copy()

# ---------- Date window + copies ----------
sd = btcdf.iloc[0, 0]
ed = btcdf.iloc[-1, 0]
print('Starting Date', sd)
print('Ending Date', ed)

# ---------- Date window + copies (use fully updated data) ----------
closedf = btcdf_updated[['Date', 'Close']].copy()
closedf['Date'] = pd.to_datetime(closedf['Date'])  # keep Timestamp
closedf_copy = closedf.copy()
print('Data coverage:', closedf['Date'].min().date(), '→', closedf['Date'].max().date())

# ---------- Scale series ----------
closedf_nodate = closedf[['Close']].copy()
scaler = MinMaxScaler(feature_range=(0,1))
series = scaler.fit_transform(closedf_nodate.values)
print(series.shape)

# ---------- Train/test split ----------
training_size = int(len(series) * 0.80)
train_data, test_data = series[0:training_size, :], series[training_size:, :1]
print('Train Data:', train_data.shape)
print('Test Data:',  test_data.shape)

def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = TIME_STEP
x_train, y_train = create_dataset(train_data, time_step)
x_test,  y_test  = create_dataset(test_data,  time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0],  x_test.shape[1],  1)
print('x_train:', x_train.shape, 'x_test:', x_test.shape)

def train_model(x_train, y_train, x_test, y_test, neurons, dropout_prob, activation, batch_size, lr):
    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(neurons, activation=activation),
        Dropout(dropout_prob),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(name='rmse'),
                           MeanAbsolutePercentageError(name='mape')])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=100, batch_size=batch_size,
                        callbacks=[early_stop], verbose=0)
    return model, history

def plot_loss_accuracy(history, title):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    rmse = history.history.get('rmse', [])
    val_rmse = history.history.get('val_rmse', [])
    mape = history.history.get('mape', [])
    val_mape = history.history.get('val_mape', [])
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(loss, label='Train Loss', color='red')
    axs[0].plot(val_loss, label='Val Loss', color='blue'); axs[0].set_title('Loss (MSE) - '+title)
    axs[1].plot(rmse, label='Train RMSE', color='green')
    axs[1].plot(val_rmse, label='Val RMSE', color='orange'); axs[1].set_title('Root Mean Squared Error - '+title)
    axs[2].plot(mape, label='Train MAPE', color='purple')
    axs[2].plot(val_mape, label='Val MAPE', color='cyan'); axs[2].set_title('Mean Absolute Percentage Error - '+title)
    for ax in axs: ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout()
    out_png = os.path.join("out", f"{title.replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluation(model, x_train, y_train, x_test, y_test, scaler):
    # Predict (scaled)
    train_pred = model.predict(x_train, verbose=0)
    test_pred  = model.predict(x_test,  verbose=0)
    # Inverse-scale to price
    train_pred_i = scaler.inverse_transform(train_pred)
    test_pred_i  = scaler.inverse_transform(test_pred)
    ytrain_i     = scaler.inverse_transform(y_train.reshape(-1,1))
    ytest_i      = scaler.inverse_transform(y_test.reshape(-1,1))
    # Safe clips for deviance metrics (require >0)
    eps = 1e-8
    ytest_pos      = np.clip(ytest_i,      eps, None)
    test_pred_pos  = np.clip(test_pred_i,  eps, None)
    ytrain_pos     = np.clip(ytrain_i,     eps, None)
    train_pred_pos = np.clip(train_pred_i, eps, None)
    def mape(y, yhat):
        mask = np.abs(y) > eps
        return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask]))) if mask.any() else float("nan")
    def smape(y, yhat): return float(2*np.mean(np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps)))
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
    results_df = pd.DataFrame(metrics)
    return results_df, ytest_i, test_pred_i

# ---------- Hyperparam sweep (single config now) ----------
neurons_set = [x_train.shape[1] * 8]
dropout_prob_set = [0.0]
activation_set = ['tanh']
batch_size_set = [32]
lr_set = [0.001]

best_mse = float('inf'); best_config = None; best_model = None
all_results = []
for neurons in neurons_set:
    for batch_size in batch_size_set:
        for dropout_prob in dropout_prob_set:
            for lr in lr_set:
                for activation in activation_set:
                    print(f"Testing: activation={activation}, neurons={neurons}, dropout={dropout_prob}, batch={batch_size}, lr={lr}")
                    start_time = time.time()
                    model, history = train_model(x_train, y_train, x_test, y_test, neurons, dropout_prob, activation, batch_size, lr)
                    training_time = time.time() - start_time
                    results_df, _, _ = evaluation(model, x_train, y_train, x_test, y_test, scaler)
                    print(results_df)
                    plot_loss_accuracy(history, title=f'{activation}-{neurons}-{dropout_prob}-{batch_size}-{lr}')
                    test_mse = results_df.loc[results_df['Metric'] == 'MSE', 'Test'].values[0]
                    if test_mse < best_mse:
                        best_mse = test_mse; best_config = {'activation':activation,'neurons':neurons,'dropout':dropout_prob,'batch_size':batch_size,'lr':lr}
                        best_model = model
                    config_result = results_df.copy()
                    config_result['activation'] = activation
                    config_result['neurons'] = neurons
                    config_result['dropout'] = dropout_prob
                    config_result['batch_size'] = batch_size
                    config_result['lr'] = lr
                    config_result['training_time_sec'] = training_time
                    all_results.append(config_result)

final_results = pd.concat(all_results, ignore_index=True)
print(f"Best Config:{best_config}")
print(f"Best Test MSE: {best_mse:.4f}")
best_model.save("out/best_btc_model.keras")           # save into out/
final_results.to_csv("out/all_model_results.csv", index=False)
print(f"final table : {final_results.head()}")

model = best_model

# ---------- One-step next-day forecast ----------
x_last = series[-time_step:].reshape(1, time_step, 1)
next_day_scaled = model.predict(x_last, verbose=0)
next_day_close_forecast = float(scaler.inverse_transform(next_day_scaled)[0,0])
print(f"[Forecast] Next-day BTC close (price): {next_day_close_forecast:.2f}")

# ---------- Optional charts (saved to out/, no fig.show()) ----------
# Train/Test fit chart
train_predict = model.predict(x_train, verbose=0)
test_predict  = model.predict(x_test,  verbose=0)
train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)

look_back = time_step
trainPredictPlot = np.empty_like(series); trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty_like(series); testPredictPlot[:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(series)-1,:] = test_predict

names = cycle(['Actual Close Price:', 'Train Predicted Close Price', 'Test Predicted Close Price'])
plotdf = pd.DataFrame({'date': closedf_copy['Date'],
                       'actual_close': closedf_copy['Close'],
                       'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                       'test_predicted_close':  testPredictPlot.reshape(1,-1)[0].tolist()})
fig = px.line(plotdf, x='date', y=['actual_close','train_predicted_close','test_predicted_close'],
              labels={'value':'BTC Price','date':'Date'})
fig.update_layout(title_text='BTC Close Price: Actual VS Prediction', font_size=15, font_color='Black', plot_bgcolor='white')
fig.for_each_trace(lambda t: t.update(name = next(names))); fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
fig.write_html("out/btc_close_actual_chart.html")
fig.write_image("out/btc_close_actual_chart.png")

# 30-day roll-ahead prediction chart (optional)
x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input = list(x_input[0]); First_output = []; n_steps = time_step; pred_days = 30; i = 0
while i < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:]).reshape((1, n_steps, 1))
    else:
        x_input = np.array(temp_input).reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)
    temp_input.extend(yhat[0].tolist()); temp_input = temp_input[1:]; First_output.extend(yhat.tolist()); i += 1
last_days = np.arange(1, time_step+1); day_pred = np.arange(time_step+1, time_step+pred_days+1)

temp_mat = np.empty((len(last_days)+pred_days+1,1)); temp_mat[:] = np.nan; temp_mat = temp_mat.reshape(1,-1).tolist()
last_actual_days_close_price = temp_mat; next_predicted_days_close_price = temp_mat
last_actual_days_close_price[0:time_step+1] = scaler.inverse_transform(series[len(series)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_close_price[time_step+1:] = scaler.inverse_transform(np.array(First_output).reshape(1,-1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_actual_days_close_price': last_actual_days_close_price,
    'next_predicted_days_close_price': next_predicted_days_close_price
})
fig2 = px.line(new_pred_plot, x=new_pred_plot.index, y=['last_actual_days_close_price','next_predicted_days_close_price'],
               labels={'value':'BTC Price','index':'Days'})
fig2.add_vline(x=time_step, line_dash="dash", line_color="black")
fig2.update_layout(title_text='LSTM BTC CLOSE PREDICTION', font_size=15, font_color='Black', plot_bgcolor='white')
fig2.update_xaxes(showgrid=False); fig2.update_yaxes(showgrid=False)
fig2.write_html("out/btc_close_prediction_chart.html")
fig2.write_image("out/btc_close_prediction_chart.png")

print(new_pred_plot.head())

# ---------- FINAL: save forecast + metrics + print ----------
def _f(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except Exception:
        return None

results_df, _, _ = evaluation(model, x_train, y_train, x_test, y_test, scaler)
metrics_test = {k: _f(v) for k, v in results_df.set_index("Metric")["Test"].items()}

last_obs_date = pd.to_datetime(closedf_copy["Date"].max())
target_close_date_utc = (last_obs_date + pd.Timedelta(days=1)).date().isoformat()
run_ts_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat()

payload = {
    "symbol": "BTCUSDT",
    "horizon": "next_daily_close",
    "run_ts_utc": run_ts_utc,
    "target_close_date_utc": target_close_date_utc,
    "forecast_close": _f(next_day_close_forecast),
    "metrics_h1": metrics_test
}

with open("out/daily_forecast.json", "w") as f:
    json.dump(payload, f, indent=2)

row = {
    "run_ts_utc": run_ts_utc,
    "target_close_date_utc": target_close_date_utc,
    "symbol": "BTCUSDT",
    "forecast_close": _f(next_day_close_forecast),
    **{f"h1_{k.lower().replace(' ', '_')}": _f(v) for k, v in metrics_test.items()}
}
hist_path = "out/history.csv"
if os.path.exists(hist_path):
    df_hist = pd.read_csv(hist_path)
    df_hist = pd.concat([df_hist, pd.DataFrame([row])], ignore_index=True)
else:
    df_hist = pd.DataFrame([row])
df_hist.to_csv(hist_path, index=False)

print(json.dumps(payload, indent=2))
print("[OK] Wrote out/daily_forecast.json and out/history.csv")
print(f"Next-day forecast close: {payload['forecast_close']:.2f}")
for k in ["RMSE","MSE","MAE","MAPE","sMAPE","R\u00b2","Explained Variance","Gamma Deviance","Poisson Deviance"]:
    if k in metrics_test and metrics_test[k] is not None:
        v = metrics_test[k]
        if k in ("MAPE","sMAPE"):
            print(f"{k}: {100*v:.2f}%")
        elif k == "R\u00b2":
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v:.2f}")
