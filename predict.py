import numpy as np
import joblib
from datetime import datetime, timedelta

import firebase_admin

from firebase_admin import credentials
from firebase_admin import db

from tensorflow.keras.models import load_model

# =========================
# FIREBASE
# =========================

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiquantrackhongkhi-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

print("Firebase Connected!")

# =========================
# LOAD MODEL
# =========================

model = load_model("air_model.keras")

print("AI Model Loaded!")

# =========================
# LOAD SCALERS
# =========================

feature_scaler = joblib.load("feature_scaler.pkl")

target_scaler = joblib.load("target_scaler.pkl")

print("Scalers Loaded!")


def calculate_nowcast(pm_list):
    if len(pm_list) < 12:
        return None
    recent = pm_list[-12:]
    c_max = max(recent)
    c_min = min(recent)
    if c_max == 0:
        return 0
    weight_factor = c_min / c_max
    if weight_factor < 0.5:
        weight_factor = 0.5

    numerator = 0
    denominator = 0
    for i in range(len(recent)):
        weight = weight_factor ** i

        numerator += recent[-(i + 1)] * weight
        denominator += weight
    return numerator / denominator


def calculate_vn_aqi(pm25):

    if pm25 is None:
        return 0

    # Chuẩn VN AQI (EPA breakpoint)
    breakpoints = [
        {"bpLow": 0.0,   "bpHigh": 12.0,   "iLow": 0,   "iHigh": 50},
        {"bpLow": 12.1,  "bpHigh": 35.4,   "iLow": 51,  "iHigh": 100},
        {"bpLow": 35.5,  "bpHigh": 55.4,   "iLow": 101, "iHigh": 150},
        {"bpLow": 55.5,  "bpHigh": 150.4,  "iLow": 151, "iHigh": 200},
        {"bpLow": 150.5, "bpHigh": 250.4,  "iLow": 201, "iHigh": 300},
        {"bpLow": 250.5, "bpHigh": 500.4,  "iLow": 301, "iHigh": 500},
    ]

    for bp in breakpoints:
        if pm25 <= bp["bpHigh"]:

            aqi = (
                (bp["iHigh"] - bp["iLow"]) /
                (bp["bpHigh"] - bp["bpLow"])
            ) * (pm25 - bp["bpLow"]) + bp["iLow"]

            return round(aqi)

    return 500
# =========================
# AI FUNCTION
# =========================

def run_ai():
    # =========================
    # READ FIREBASE HISTORY
    # =========================

    ref = db.reference("history")

    data = ref.get()

    values = []

    for item in data.values():

        try:
            datetime.strptime(
                item['time'],
                "%Y-%m-%d %H:%M:%S"
            )
            values.append(item)

        except:
            continue

    values.sort(
        key=lambda x:
        datetime.strptime(
            x['time'],
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print("Total Records:", len(values))

    # lấy 10 dữ liệu mới nhất
    last_10 = values[-10:]

    # =========================
    # LAST HISTORY TIME
    # =========================

    last_history_time = datetime.strptime(
        last_10[-1]['time'],
        "%Y-%m-%d %H:%M:%S"
    )

    # =========================
    # CREATE INPUT DATA
    # =========================

    X_data = []

    for item in last_10:

        # parse time
        dt = datetime.strptime(item['time'], "%Y-%m-%d %H:%M:%S")

        row = [
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            item['pm25'],
            item['temp'],
            item['humi']
        ]

        X_data.append(row)

    # convert sang numpy
    X = np.array([X_data], dtype=np.float32)

    print("Input Shape:", X.shape)

    # =========================
    # NORMALIZE
    # =========================

    X_reshaped = X.reshape(-1, 9)

    X_scaled = feature_scaler.transform(X_reshaped)

    X_scaled = X_scaled.reshape(1, 10, 9)

    # =========================
    # FUTURE FORECAST
    # =========================

    future_predictions = []

    current_sequence = X_scaled.copy()

    pm25_values = [float(x['pm25']) for x in values]

    forecast_pm25_list = pm25_values.copy()

    for i in range(12):

        # predict
        prediction_scaled = model.predict(current_sequence, verbose=0)

        # inverse transform
        prediction = target_scaler.inverse_transform(prediction_scaled)

        pred_pm25 = float(prediction[0][0])
        pred_temp = float(prediction[0][1])
        pred_humi = float(prediction[0][2])

        # =========================
        # ADD PM2.5 TO NOWCAST LIST
        # =========================
        forecast_pm25_list.append(pred_pm25)

        # =========================
        # NOWCAST
        # =========================
        forecast_nowcast = calculate_nowcast(
            forecast_pm25_list
        )

        # =========================
        # VN_AQI
        # =========================
        forecast_vn_aqi = calculate_vn_aqi(
            forecast_nowcast if forecast_nowcast is not None else 0
        )
        # thời gian tương lai
        future_time = last_history_time + timedelta(hours=i + 1)

        # lưu prediction
        future_predictions.append({

            "time": future_time.strftime("%Y-%m-%d %H:%M:%S"),

            "pm25": round(pred_pm25, 2),
            "temp": round(pred_temp, 2),
            "humi": round(pred_humi, 2),

            # 🔥 NOWCAST + VN_AQI
            "nowcast_pm25": round(forecast_nowcast, 2)
                if forecast_nowcast else 0,

            "vn_aqi": forecast_vn_aqi
        })

        # tạo input mới
        next_input = np.array([
            future_time.year,
            future_time.month,
            future_time.day,
            future_time.hour,
            future_time.minute,
            future_time.second,
            pred_pm25,
            pred_temp,
            pred_humi
        ], dtype=np.float32)

        # normalize
        next_input_scaled = feature_scaler.transform(
            next_input.reshape(1, -1)
        )

        # reshape
        next_input_scaled = next_input_scaled.reshape(1, 1, 9)

        # update sequence
        current_sequence = np.concatenate([
            current_sequence[:, 1:, :],
            next_input_scaled
        ], axis=1)

    # prediction đầu tiên
    pred_pm25 = future_predictions[0]["pm25"]
    pred_temp = future_predictions[0]["temp"]
    pred_humi = future_predictions[0]["humi"]

    print("\n===== 12-HOUR FORECAST =====")

    for item in future_predictions:

        print(
        item["time"],
        "| PM2.5:", item["pm25"],
        "| Temp:", item["temp"],
        "| Humi:", item["humi"]
        )

    # =========================
    # EXTRACT VALUES
    # =========================
    pm25_values = [float(x['pm25']) for x in values]
    nowcast_pm25 = calculate_nowcast(pm25_values)

    vn_aqi = calculate_vn_aqi(nowcast_pm25)
    temp_values = [float(x['temp']) for x in values]
    humi_values = [float(x['humi']) for x in values]

    # =========================
    # CURRENT VALUES
    # =========================

    current_pm25 = pm25_values[-1]
    current_temp = temp_values[-1]
    current_humi = humi_values[-1]

    # =========================
    # FIRST FORECAST VALUES
    # =========================

    first_forecast = future_predictions[0]

    forecast_pm25 = first_forecast["pm25"]
    forecast_temp = first_forecast["temp"]
    forecast_humi = first_forecast["humi"]

    # =========================
    # CHANGE VALUE
    # =========================

    pm25_change = forecast_pm25 - current_pm25
    temp_change = forecast_temp - current_temp
    humi_change = forecast_humi - current_humi

    # =========================
    # TREND RESULT
    # =========================

    def get_trend(change, threshold):

        if change > threshold:
            return "tăng"

        elif change < -threshold:
            return "giảm"

        return "ổn định"

    pm25_trend = get_trend(pm25_change, 3)
    temp_trend = get_trend(temp_change, 0.5)
    humi_trend = get_trend(humi_change, 2)

    # =========================
    # 12-HOUR ANALYSIS
    # =========================

    forecast_pm25_avg = sum(
        item["pm25"] for item in future_predictions
    ) / len(future_predictions)

    forecast_temp_avg = sum(
        item["temp"] for item in future_predictions
    ) / len(future_predictions)

    forecast_humi_avg = sum(
        item["humi"] for item in future_predictions
    ) / len(future_predictions)

    pm25_12h_change = (
        forecast_pm25_avg - current_pm25
    )

    temp_12h_change = (
        forecast_temp_avg - current_temp
    )

    humi_12h_change = (
        forecast_humi_avg - current_humi
    )

    # =========================
    # AI SMART ADVICE
    # =========================

    advice = []

    avg_aqi = sum(
        item["vn_aqi"] for item in future_predictions
    ) / len(future_predictions)

    max_aqi = max(
        item["vn_aqi"] for item in future_predictions
    )

    # =========================
    # AQI SUMMARY
    # =========================

    if avg_aqi <= 50:

        advice.append(
            "Chất lượng không khí 12 giờ tới ở mức tốt."
        )

    elif avg_aqi <= 100:

        advice.append(
            "Không khí 12 giờ tới ở mức trung bình."
        )

    elif avg_aqi <= 150:

        advice.append(
            "Không khí có xu hướng kém trong 12 giờ tới."
        )

    elif avg_aqi <= 200:

        advice.append(
            "Không khí ở mức xấu, nên hạn chế ra ngoài."
        )

    else:

        advice.append(
            "Ô nhiễm không khí nghiêm trọng trong 12 giờ tới."
        )

    # =========================
    # PM2.5 12H ADVICE
    # =========================

    if pm25_12h_change > 5:
        advice.append(
            f"PM2.5 trong 12 giờ tới có xu hướng tăng trung bình "
            f"{abs(pm25_12h_change):.1f} µg/m³ → chất lượng không khí xấu dần."
        )

    elif pm25_12h_change < -5:
        advice.append(
            f"PM2.5 trong 12 giờ tới có xu hướng giảm trung bình "
            f"{abs(pm25_12h_change):.1f} µg/m³ → không khí cải thiện."
        )

    else:
        advice.append("PM2.5 trong 12 giờ tới duy trì ổn định.")

    # =========================
    # TEMP 12H ADVICE
    # =========================

    if temp_12h_change > 1:
        advice.append(
            f"Nhiệt độ trung bình 12 giờ tới tăng khoảng "
            f"{abs(temp_12h_change):.1f}°C."
        )

    elif temp_12h_change < -1:
        advice.append(
            f"Nhiệt độ trung bình 12 giờ tới giảm khoảng "
            f"{abs(temp_12h_change):.1f}°C."
        )

    else:
        advice.append("Nhiệt độ 12 giờ tới ổn định.")

    # =========================
    # HUMIDITY 12H ADVICE
    # =========================

    if humi_12h_change > 3:
        advice.append(
            f"Độ ẩm trung bình 12 giờ tới tăng khoảng "
            f"{abs(humi_12h_change):.1f}%."
        )

    elif humi_12h_change < -3:
        advice.append(
            f"Độ ẩm trung bình 12 giờ tới giảm khoảng "
            f"{abs(humi_12h_change):.1f}%."
        )

    else:
        advice.append("Độ ẩm 12 giờ tới ổn định.")

    # =========================
    # EXTREME WARNING
    # =========================

    if max_aqi >= 200:

        advice.append(
            "Có thời điểm chất lượng không khí ở mức rất xấu."
        )

    # =========================
    # FINAL TEXT
    # =========================

    advice_text = " ".join(advice)

    # =========================
    # PRINT TREND
    # =========================

    print("\n===== TREND ANALYSIS =====")

    print(f"PM2.5 {pm25_trend}: {abs(pm25_change):.2f} µg/m³")
    print(f"Temp {temp_trend}: {abs(temp_change):.2f}°C")
    print(f"Humi {humi_trend}: {abs(humi_change):.2f}%")

    print("\n===== AI ADVICE =====")

    print(advice_text)

    # =========================
    # SAVE CURRENT PREDICTION
    # =========================

    prediction_ref = db.reference("prediction")

    prediction_ref.set({

        "pm25": round(pred_pm25, 2),
        "temp": round(pred_temp, 2),
        "humi": round(pred_humi, 2),

        # =========================
        # NOWCAST + VN_AQI
        # =========================
        "nowcast_pm25": round(nowcast_pm25, 2) if nowcast_pm25 else 0,
        "vn_aqi": vn_aqi,

        "pm25_trend": pm25_trend,
        "temp_trend": temp_trend,
        "humi_trend": humi_trend,

        "pm25_change": round(pm25_change, 2),
        "temp_change": round(temp_change, 2),
        "humi_change": round(humi_change, 2),

        "advice": advice_text,

        "time": future_predictions[0]["time"]
    })

    # =========================
    # SAVE 12-HOUR FORECAST
    # =========================

    forecast_ref = db.reference("forecast")

    forecast_ref.set(future_predictions)

    print("\n12-hour forecast saved!")

    print("\nPrediction saved to Firebase!")

# =========================
# FIREBASE LISTENER
# =========================

history_ref = db.reference("history")
first_run = True

def listener(event):
  global first_run

  # bỏ lần load đầu tiên
  if first_run:
    first_run = False
    print("Firebase listener started.")
    return
  print("\nNew Firebase data detected!")
  try:
    run_ai()
  except Exception as e:
    print("AI Error:", e)

run_ai()
# lắng nghe realtime
history_ref.listen(listener)

print("Listening for Firebase changes...")