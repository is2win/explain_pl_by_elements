import math

def compute_spot_delta(input_data, real_deltas=None):
    # Извлекаем данные
    pv_yesterday = input_data.get('PV_yesterday', 0.0)
    pv_today = input_data.get('PV_today', 0.0)
    spots = input_data.get('spots', {})

    delta_pv = pv_today - pv_yesterday

    # Сбор изменений спот-цен
    assets = list(spots.keys())
    delta_s = {}
    abs_delta_s_sum = 0.0
    for asset, prices in spots.items():
        s_yesterday = prices.get('yesterday', 0.0)
        s_today = prices.get('today', 0.0)
        delta = s_today - s_yesterday
        delta_s[asset] = delta
        abs_delta_s_sum += abs(delta)

    # Шаг 2: Расчёт spot_delta
    spot_deltas = {}
    total_spot_delta = 0.0
    if real_deltas:
        # Если предоставлены реальные дельты, используем их
        for asset in assets:
            delta_i = real_deltas.get(asset, 0.0)
            contribution = delta_i * delta_s[asset]
            spot_deltas[asset] = contribution
            total_spot_delta += contribution
    else:
        # Аппроксимация, как раньше
        if abs_delta_s_sum == 0:
            spot_deltas = {asset: 0.0 for asset in assets}
        else:
            for asset in assets:
                delta = delta_s[asset]
                if delta == 0:
                    spot_deltas[asset] = 0.0
                    continue
                weight = abs(delta) / abs_delta_s_sum
                delta_i = (delta_pv * weight) / delta
                contribution = delta_i * delta
                spot_deltas[asset] = contribution
                total_spot_delta += contribution
    # В аппроксимации total_spot_delta должен быть равен delta_pv, но на всякий случай считаем
    if not real_deltas:
        total_spot_delta = delta_pv  # По конструкции равно

    # Шаг 3: Нелинейный остаток
    nonlinear_remainder = delta_pv - total_spot_delta

    # Шаг 4: Сумма произведений ΔS_i × ΔS_j для всех уникальных пар i < j
    pair_sum = 0.0
    n = len(assets)
    for i in range(n):
        for j in range(i+1, n):
            pair_sum += delta_s[assets[i]] * delta_s[assets[j]]

    # Шаг 5: Spot cross gamma
    if pair_sum != 0:
        spot_cross_gamma = nonlinear_remainder / pair_sum
    else:
        spot_cross_gamma = 0.0  # Избегаем деления на ноль

    # Шаг 6: Residual (по буквальной формуле из описания)
    residual = delta_pv - total_spot_delta - spot_cross_gamma

    # Возврат результата
    result = {
        'spot_deltas': spot_deltas,
        'total_spot_delta': total_spot_delta,
        'nonlinear_remainder': nonlinear_remainder,
        'pair_sum': pair_sum,
        'spot_cross_gamma': spot_cross_gamma,
        'residual': residual
    }
    return result

# Пример использования
if __name__ == "__main__":
    example_input = {
        "PV_yesterday": 1000.0,
        "PV_today": 1100.0,
        "spots": {
            "Sber": {"yesterday": 200.0, "today": 210.0},
            "VTB": {"yesterday": 160.0, "today": 148.0},
            "Gazprom": {"yesterday": 300.0, "today": 305.0}
        }
    }
    result = compute_spot_delta(example_input)
    print(result)

    # Пример с реальными дельтами (если есть)
    real_deltas_example = {"Sber": 5.0, "VTB": -3.0, "Gazprom": 4.0}
    result_with_real = compute_spot_delta(example_input, real_deltas=real_deltas_example)
    print(result_with_real) 