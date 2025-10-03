# notify_telegram.py
import os, json, requests

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

with open("out/daily_forecast.json", "r") as f:
    data = json.load(f)

m = data.get("metrics_h1") or {}
parts = []
for key in ["RMSE","MSE","MAE","MAPE","sMAPE","R²","Explained Variance","Gamma Deviance","Poisson Deviance"]:
    if key in m:
        val = m[key]
        if key in ["MAPE","sMAPE"]:
            parts.append(f"{key} {100*val:.2f}%")
        elif key == "R²":
            parts.append(f"R² {val:.3f}")
        else:
            parts.append(f"{key} {val:.2f}")
metric_line = "\n" + " | ".join(parts) if parts else ""

msg = (
    "📈 Daily BTC Forecast\n"
    f"Run (UTC): {data['run_ts_utc']}\n"
    f"Target Close (UTC date): {data['target_close_date_utc']}\n"
    f"Symbol: {data['symbol']}\n"
    f"Forecast Close: {data['forecast_close']:.2f}"
    f"{metric_line}"
)

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
r.raise_for_status()
print("[OK] Telegram sent.")
