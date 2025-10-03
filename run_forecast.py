#!/usr/bin/env python3
# coding: utf-8

# ---------------- Headless plotting (CI safe) ----------------
import matplotlib
matplotlib.use("Agg")

# ---------------- Imports ----------------
import os, math, time, json, datetime as dt
import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, explained_variance_score,
    r2_score, mean_poisson_deviance, mean_gamma_deviance
)

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError

# ---------------- Config ----------------
os.makedirs("out", exist_ok=True)

TIME_STEP      = int(os.getenv("TIME_STEP", "15"))
DATA_SRC       = os.getenv("DATA_SOURCE", "binance").lower()   # binance|coingecko|yfinance
SYMBOL         = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL       = os.getenv("INTERVAL", "1d")
HISTORY_START  = os.getenv("HISTORY_START", "2020-10-01")      # UTC

print(f"[CFG] TIME_STEP={TIME_STEP}  DATA_SOURCE={DATA_SRC}  SYMBOL={SYMBOL}  START={HISTORY_START}")

# ---------------- Data fetchers ----------------
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def _df_from_binance_rows(rows):
    if not rows:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(rows, columns=[
        "Open Time","Open","High","Low","Close","Volume",
        "Close Time","Quote Volume","Number of Trades",
        "Taker Buy Base","Taker Buy Quote","Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    df = df[["Date","Open","High","Low","Close","Volume"]].copy()
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
    return df.sort_values("Date").reset_index(drop=True)

def fetch_all_binance(symbol="BTCUSDT", interval="1d", start="2020-10-01", pause=0.15):
    """Paginate Binance (limit=1000) from start → now."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    all_rows, page = [], 0
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": start_ms}
        r = requests.get(BINANCE_URL, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text[:200]}")
        batch = r.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_rows.extend(batch)
        page += 1
        last_close_ms = batch[-1][6]
        last_open_ts  = pd.to_datetime(batch[-1][0], unit="ms", utc=True)
        print(f"[binance] page {page}: {len(batch)} rows → {last_open_ts.date()}")
        next_start = last_close_ms + 1
        if next_start <= start_ms:  # safety
            break
        start_ms = next_start
        if len(batch) < 1000:
            break
        if pause:
            time.sleep(pause)
    return _df_from_binance_rows(all_rows)

def fetch_from_coingecko(start="2020-10-01"):
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts   = int(pd.Timestamp.utcnow().timestamp())
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {"vs_currency":"usd", "from":start_ts, "to":end_ts}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    prices = js.get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(prices, columns=["ts","Close"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["Open","High","Low"]:
        df[col] = df["Close"]
    df["Volume"] = np.nan
    df = df[["Date","Open","High","Low","Close","Volume"]]
    return df.sort_values("Date").reset_index(drop=True)

def fetch_from_yf(start="2020-10-01"):
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    t = yf.download("BTC-USD",
                    start=pd.Timestamp(start, tz="UTC").tz_convert(None).date(),
                    end=pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None).date(),
                    interval="1d",
                    progress=False)
    if t is None or t.empty:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    t = t.reset_index().rename(columns=str.title)
    t["Date"] = pd.to_datetime(t["Date"], utc=True)
    t = t[["Date","Open","High","Low","Close","Volume"]]
    return t.sort_values("Date").reset_index(drop=True)

def fetch_history(symbol=SYMBOL, interval=INTERVAL, start=HISTORY_START):
    """Fetch full history from chosen DATA_SRC with fallbacks."""
    try:
        if DATA_SRC == "binance":
            df = fetch_all_binance(symbol, interval, start)
            if df.empty:
                print("[WARN] Binance returned no rows; falling back to CoinGecko…")
                df = fetch_from_coingecko(start)
        elif DATA_SRC == "coingecko":
            df = fetch_from_coingecko(start)
        elif DATA_SRC == "yfinance":
            df = fetch_from_yf(start)
        else:
            print(f"[WARN] Unknown DATA_SOURCE={DATA_SRC}; using CoinGecko")
            df = fetch_from_coingecko(start)
    except Exception as e:
        print("[WARN] Primary fetch failed:", e, "→ trying CoinGecko then yfinance…")
        df = fetch_from_coingecko(start)
        if df.empty:
            df = fetch_from_yf(start)
    return df

# ---------------- Fetch data (no CSV needed) ----------------
print(f"[DATA] Fetching {SYMBOL} {INTERVAL} from {HISTORY_START} → now via {DATA_SRC}…")
btcdf_updated = fetch_history()
if btcdf_updated.empty:
    raise RuntimeError("No price data fetched from any provider.")

print("[INFO] Coverage:", btcdf_updated["Date"].iloc[0], "→", btcdf_updated["Date"].iloc[-1])
btcdf_updated.to_csv("out/btc_updated.csv", index=False)  # optional snapshot

# ---------------- Build modeling frame ----------------
closedf = btcdf_updated[["Date","Close"]].copy()
closedf["Date"] = pd.to_datetime(closedf["Date"], utc=True)
closedf_copy = closedf.copy()
print("Data coverage:", closedf["Date"].min().date(), "→", closedf["Date"].max().date())

# ---------------- Scale series ----------------
scaler = MinMaxScaler(feature_range=(0,1))
series = scaler.fit_transform(closedf[["Close"]].values)
print("[INFO] Series shape:", series.shape)

# ---------------- Train/test split & windows ----------------
training_size = int(len(series) * 0.80)
train_data, test_data = series[:training_size], series[training_size:]
print("Train:", train_data.shape, "Test:", test_data.shape)

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
print("x_train:", x_train.shape, "x_test:", x_test.shape)

# ---------------- Model / training ----------------
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
    early = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     epochs=100, batch_size=batch_size,
                     callbacks=[early], verbose=0)
    return model, hist

def plot_loss_accuracy(history, title):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    rmse = history.history.get('rmse', [])
    val_rmse = history.history.get('val_rmse', [])
    mape = history.history.get('mape', [])
    val_mape = history.history.get('val_mape', [])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(loss, label='Train Loss', color='red');   axs[0].plot(val_loss, label='Val Loss', color='blue')
    axs[0].set_title('Loss (MSE) - ' + title); axs[0].legend()
    axs[1].plot(rmse, label='Train RMSE', color='green'); axs[1].plot(val_rmse, label='Val RMSE', color='orange')
    axs[1].set_title('RMSE - ' + title); axs[1].legend()
    axs[2].plot(mape, label='Train MAPE', color='purple'); axs[2].plot(val_mape, label='Val MAPE', color='cyan')
    axs[2].set_title('MAPE - ' + title); axs[2].legend()
    for ax in axs: ax.set_xlabel('Epoch')
    plt.tight_layout()
    out_png = os.path.join("out", f"{title.replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluation(model, x_train, y_train, x_test, y_test, scaler):
    train_pred = model.predict(x_train, verbose=0)
    test_pred  = model.predict(x_test,  verbose=0)
    train_pred_i = scaler.inverse_transform(train_pred)
    test_pred_i  = scaler.inverse_transform(test_pred)
    ytrain_i     = scaler.inverse_transform(y_train.reshape(-1,1))
    ytest_i      = scaler.inverse_transform(y_test.reshape(-1,1))

    eps = 1e-8
    def mape(y, yhat):
        mask = np.abs(y) > eps
        return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask]))) if mask.any() else float("nan")
    def smape(y, yhat): return float(2*np.mean(np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps)))

    ytest_pos      = np.clip(ytest_i,      eps, None)
    test_pred_pos  = np.clip(test_pred_i,  eps, None)
    ytrain_pos     = np.clip(ytrain_i,     eps, None)
    train_pred_pos = np.clip(train_pred_i, eps, None)

    metrics = {
        "Metric": ["RMSE","MSE","MAE","MAPE","sMAPE","R\u00b2","Explained Variance","Gamma Deviance","Poisson Deviance"],
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
    return pd.DataFrame(metrics), ytest_i, test_pred_i

# single-config sweep
neurons_set      = [x_train.shape[1] * 8]
dropout_prob_set = [0.0]
activation_set   = ['tanh']
batch_size_set   = [32]
lr_set           = [0.001]

best_mse = float('inf'); best_model = None; all_results=[]
for neurons in neurons_set:
    for batch_size in batch_size_set:
        for dropout_prob in dropout_prob_set:
            for lr in lr_set:
                for activation in activation_set:
                    tag = f'{activation}-{neurons}-{dropout_prob}-{batch_size}-{lr}'
                    print(f"[TRAIN] {tag}")
                    t0=time.time()
                    model, history = train_model(x_train, y_train, x_test, y_test, neurons, dropout_prob, activation, batch_size, lr)
                    results_df, _, _ = evaluation(model, x_train, y_train, x_test, y_test, scaler)
                    plot_loss_accuracy(history, title=tag)
                    test_mse = results_df.loc[results_df['Metric']=='MSE','Test'].values[0]
                    if test_mse < best_mse:
                        best_mse = test_mse; best_model = model
                    df = results_df.copy()
                    df["tag"]=tag; df["training_time_sec"]=time.time()-t0
                    all_results.append(df)

final_results = pd.concat(all_results, ignore_index=True)
best_model.save("out/best_btc_model.keras")
final_results.to_csv("out/all_model_results.csv", index=False)
print("[INFO] Best Test MSE:", best_mse)

model = best_model

# ---------------- One-step next-day forecast ----------------
x_last = series[-time_step:].reshape(1, time_step, 1)
next_day_scaled = model.predict(x_last, verbose=0)
next_day_close_forecast = float(scaler.inverse_transform(next_day_scaled)[0,0])
print(f"[Forecast] Next-day close: {next_day_close_forecast:.2f}")

# ---------------- Optional charts (saved to out/) ----------------
train_pred = scaler.inverse_transform(model.predict(x_train, verbose=0))
test_pred  = scaler.inverse_transform(model.predict(x_test,  verbose=0))

look_back = time_step
trainPredictPlot = np.empty_like(series); trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_pred)+look_back,:] = train_pred

testPredictPlot = np.empty_like(series); testPredictPlot[:] = np.nan
testPredictPlot[len(train_pred)+(look_back*2)+1:len(series)-1,:] = test_pred

names = cycle(['Actual Close Price:', 'Train Predicted Close Price', 'Test Predicted Close Price'])
plotdf = pd.DataFrame({
    'date': closedf_copy['Date'].dt.tz_convert('UTC'),
    'actual_close': closedf_copy['Close'],
    'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
    'test_predicted_close':  testPredictPlot.reshape(1,-1)[0].tolist()
})
fig = px.line(plotdf, x='date', y=['actual_close','train_predicted_close','test_predicted_close'],
              labels={'value':'BTC Price','date':'Date'})
fig.update_layout(title_text='BTC Close Price: Actual VS Prediction', font_size=15, font_color='Black', plot_bgcolor='white')
fig.for_each_trace(lambda t: t.update(name = next(names))); fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
fig.write_html("out/btc_close_actual_chart.html")
try:
    fig.write_image("out/btc_close_actual_chart.png")  # needs kaleido
except Exception as e:
    print("[WARN] PNG save skipped (kaleido not available):", e)

# 30-day roll-ahead (optional)
x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input[0]); First_output=[]; n_steps=time_step; pred_days=30
for _ in range(pred_days):
    if len(temp_input)>time_step:
        x_in = np.array(temp_input[1:]).reshape((1, n_steps, 1))
    else:
        x_in = np.array(temp_input).reshape((1, n_steps, 1))
    yhat = model.predict(x_in, verbose=0)
    temp_input.extend(yhat[0].tolist()); temp_input=temp_input[1:]; First_output.extend(yhat.tolist())

temp_mat = np.empty((len(range(1,time_step+1))+pred_days+1,1)); temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()
last_actual_days_close_price  = temp_mat
next_predicted_days_close_price = temp_mat
last_actual_days_close_price[0:time_step+1] = scaler.inverse_transform(series[len(series)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_close_price[time_step+1:] = scaler.inverse_transform(np.array(First_output).reshape(1,-1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_actual_days_close_price': last_actual_days_close_price,
    'next_predicted_days_close_price': next_predicted_days_close_price
})
fig2 = px.line(new_pred_plot, x=new_pred_plot.index,
               y=['last_actual_days_close_price','next_predicted_days_close_price'],
               labels={'value':'BTC Price','index':'Days'})
fig2.add_vline(x=time_step, line_dash="dash", line_color="black")
fig2.update_layout(title_text='LSTM BTC CLOSE PREDICTION', font_size=15, font_color='Black', plot_bgcolor='white')
fig2.update_xaxes(showgrid=False); fig2.update_yaxes(showgrid=False)
fig2.write_html("out/btc_close_prediction_chart.html")
try:
    fig2.write_image("out/btc_close_prediction_chart.png")
except Exception as e:
    print("[WARN] PNG save skipped (kaleido not available):", e)

# ---------------- Final JSON + CSV ----------------
def _f(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except Exception:
        return None

results_df, _, _ = evaluation(model, x_train, y_train, x_test, y_test, scaler)
metrics_test = {k: _f(v) for k, v in results_df.set_index("Metric")["Test"].items()}

last_obs_date = closedf_copy["Date"].max()  # tz-aware
target_close_date_utc = (last_obs_date + pd.Timedelta(days=1)).date().isoformat()
run_ts_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat()

payload = {
    "symbol": SYMBOL,
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
    "symbol": SYMBOL,
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
for k, v in metrics_test.items():
    if v is None: continue
    if k in ("MAPE","sMAPE"): print(f"{k}: {100*v:.2f}%")
    elif k == "R\u00b2":      print(f"{k}: {v:.3f}")
    else:                    print(f"{k}: {v:.2f}")
