import math

def compute_spot_delta(input_data):
    # Извлекаем данные
    pv_yesterday = input_data.get('PV_yesterday', 0.0)
    pv_today = input_data.get('PV_today', 0.0)
    spots = input_data.get('spots', {})

    delta_pv = pv_today - pv_yesterday

    # Сбор изменений спот-цен
    delta_s = {}
    abs_delta_s_sum = 0.0
    for asset, prices in spots.items():
        s_yesterday = prices.get('yesterday', 0.0)
        s_today = prices.get('today', 0.0)
        delta = s_today - s_yesterday
        delta_s[asset] = delta
        abs_delta_s_sum += abs(delta)

    if abs_delta_s_sum == 0:
        return {asset: 0.0 for asset in spots}  # Нет изменений, дельты нулевые

    # Расчёт вклада для каждого актива
    spot_deltas = {}
    for asset, delta in delta_s.items():
        if delta == 0:
            spot_deltas[asset] = 0.0
            continue
        weight = abs(delta) / abs_delta_s_sum
        delta_i = (delta_pv * weight) / delta  # Это (ΔPV * |ΔS_i| / Sum) / ΔS_i = ΔPV * sign(ΔS_i) / Sum
        contribution = delta_i * delta
        spot_deltas[asset] = contribution

    return spot_deltas

# Пример использования
if __name__ == "__main__":
    example_input = {
        "PV_yesterday": 1000.0,
        "PV_today": 1050.0,
        "spots": {
            "Sber": {"yesterday": 200.0, "today": 205.0},
            "VTB": {"yesterday": 150.0, "today": 148.0},
            "Gazprom": {"yesterday": 300.0, "today": 310.0},
        }
    }
    result = compute_spot_delta(example_input)
    print(result) 