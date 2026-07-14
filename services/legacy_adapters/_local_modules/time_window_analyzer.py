import pandas as pd
from datetime import time


def _is_empty(value):
    return pd.isna(value) or value == "" or value is None


def _time_to_minutes(value):
    if _is_empty(value):
        return None

    if isinstance(value, time):
        return value.hour * 60 + value.minute

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None

        numeric = float(value)

        # 엑셀 시간값: 0.3333 = 08:00, 0.9166 = 22:00
        if 0 <= numeric < 1:
            return int(round(numeric * 24 * 60))

        # 8.5 = 08:30
        if 1 <= numeric < 24:
            hour = int(numeric)
            minute = int(round((numeric - hour) * 60))
            return hour * 60 + minute

        # 이미 분 단위로 들어온 값
        return int(round(numeric))

    text = str(value).strip()

    if ":" in text:
        try:
            hour, minute = text.split(":")[:2]
            return int(float(hour)) * 60 + int(float(minute))
        except Exception:
            return None

    try:
        numeric = float(text)

        if 0 <= numeric < 1:
            return int(round(numeric * 24 * 60))

        if 1 <= numeric < 24:
            hour = int(numeric)
            minute = int(round((numeric - hour) * 60))
            return hour * 60 + minute

        return int(round(numeric))
    except Exception:
        return None


def minutes_to_time_text(minutes):
    if minutes is None or pd.isna(minutes):
        return "-"

    try:
        minutes = int(minutes)
    except Exception:
        return "-"

    minutes = minutes % (24 * 60)
    hour = minutes // 60
    minute = minutes % 60

    return f"{hour:02d}:{minute:02d}"


def _parse_available_time(value, default_value):
    parsed = _time_to_minutes(value)

    if parsed is None:
        return default_value

    return parsed


def _get_travel_time(row):
    time_columns = [
        "travel_time_min",
        "time_min",
        "duration_min",
        "moving_time_min",
    ]

    for col in time_columns:
        if col in row.index:
            value = pd.to_numeric(row[col], errors="coerce")

            if pd.notna(value):
                return float(value)

    distance_columns = [
        "distance_km",
        "network_distance_km",
    ]

    for col in distance_columns:
        if col in row.index:
            value = pd.to_numeric(row[col], errors="coerce")

            if pd.notna(value):
                distance = float(value)

                # 평균 속도 40km/h 가정
                estimated_time = distance * 1.5

                # 최소 5분
                return max(estimated_time, 5)

    return None


def _get_target_store_name(row):
    possible_columns = [
        "retailer_name",
        "target_store",
        "to_store",
        "store_name",
    ]

    for col in possible_columns:
        if col in row.index and not _is_empty(row[col]):
            return row[col]

    return None


def analyze_trade_time_windows(cutline_result, stores, departure_time):
    if cutline_result is None or cutline_result.empty:
        return pd.DataFrame(), "거래가능시간을 판별할 경로 데이터가 없습니다."

    if stores is None or stores.empty:
        return pd.DataFrame(), "stores 데이터가 없습니다."

    if "store_name" not in stores.columns:
        return pd.DataFrame(), "stores 시트에 store_name 열이 필요합니다."

    if "available_start" not in stores.columns or "available_end" not in stores.columns:
        return (
            pd.DataFrame(),
            "stores 시트에 available_start, available_end 열이 필요합니다.",
        )

    departure_min = _time_to_minutes(departure_time)

    if departure_min is None:
        departure_min = 9 * 60

    store_time_map = {}

    for _, store_row in stores.iterrows():
        store_name = store_row["store_name"]

        start_min = _parse_available_time(
            store_row.get("available_start"),
            0,
        )

        end_min = _parse_available_time(
            store_row.get("available_end"),
            24 * 60 - 1,
        )

        store_time_map[store_name] = {
            "available_start_min": start_min,
            "available_end_min": end_min,
            "available_start": minutes_to_time_text(start_min),
            "available_end": minutes_to_time_text(end_min),
        }

    result = cutline_result.copy()

    departure_times = []
    travel_times = []
    arrival_times = []
    available_starts = []
    available_ends = []
    time_statuses = []
    final_statuses = []
    time_reasons = []

    for _, row in result.iterrows():
        target_store_name = _get_target_store_name(row)
        travel_time = _get_travel_time(row)

        departure_times.append(minutes_to_time_text(departure_min))
        travel_times.append(travel_time)

        if target_store_name not in store_time_map:
            arrival_times.append("-")
            available_starts.append("-")
            available_ends.append("-")
            time_statuses.append("확인 불가")
            final_statuses.append("불가능")
            time_reasons.append("대상 점포의 거래가능시간 정보를 찾을 수 없습니다.")
            continue

        store_time = store_time_map[target_store_name]

        available_start_min = store_time["available_start_min"]
        available_end_min = store_time["available_end_min"]

        available_starts.append(store_time["available_start"])
        available_ends.append(store_time["available_end"])

        if travel_time is None or pd.isna(travel_time):
            arrival_times.append("-")
            time_statuses.append("확인 불가")
            final_statuses.append("불가능")
            time_reasons.append("이동시간 데이터가 없어 도착시간을 계산할 수 없습니다.")
            continue

        arrival_min = departure_min + travel_time
        arrival_text = minutes_to_time_text(arrival_min)

        arrival_times.append(arrival_text)

        arrival_min_day = int(arrival_min) % (24 * 60)

        if available_start_min <= available_end_min:
            is_available = available_start_min <= arrival_min_day <= available_end_min
        else:
            is_available = (
                arrival_min_day >= available_start_min
                or arrival_min_day <= available_end_min
            )

        if is_available:
            time_status = "가능"
            reason = "도착 예정 시간이 거래가능시간 안에 있습니다."
        else:
            time_status = "시간 불가"
            reason = "도착 예정 시간이 거래가능시간을 벗어났습니다."

        cutline_status = str(row.get("cutline_status", ""))

        if "불가능" in cutline_status or "초과" in cutline_status:
            final_status = "불가능"
            reason = "거리 컷라인 조건을 만족하지 못합니다."
        elif time_status == "가능":
            final_status = "가능"
        else:
            final_status = "불가능"

        time_statuses.append(time_status)
        final_statuses.append(final_status)
        time_reasons.append(reason)

    result["departure_time"] = departure_times
    result["travel_time_min"] = travel_times
    result["arrival_time"] = arrival_times
    result["available_start"] = available_starts
    result["available_end"] = available_ends
    result["time_status"] = time_statuses
    result["final_status"] = final_statuses
    result["time_reason"] = time_reasons

    return result, None
