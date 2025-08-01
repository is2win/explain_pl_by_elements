# Explain PL by Elements

## Описание проекта
Этот проект реализует методологию распределения изменения стоимости деривативного контракта по компонентам

## Шаги расчёта
Функция `compute_spot_delta` теперь вычисляет не только spot_delta, но и spot_cross_gamma и residual:

1. **Вычисление общего изменения PV**: ΔPV = PV_сегодня - PV_вчера.

2. **Spot delta**: Аппроксимация вклада по активам пропорционально |ΔS_i| или использование реальных дельт (если переданы).

3. **Нелинейный остаток**: remainder = ΔPV - total_spot_delta.

4. **Сумма парных произведений**: pair_sum = ∑_{i<j} ΔS_i × ΔS_j.

5. **Spot cross gamma**: gamma ≈ remainder / pair_sum (если pair_sum ≠ 0).

6. **Residual**: residual = ΔPV - total_spot_delta - gamma.

## Формат ввода данных
Не изменился, но добавлен опциональный параметр real_deltas (словарь {asset: delta_i}).

## Пример использования
```python
example_input = {
    "PV_yesterday": 1000.0,
    "PV_today": 1100.0,
    "spots": {
        "Sber": {"yesterday": 200.0, "today": 210.0},
        "VTB": {"yesterday": 150.0, "today": 148.0},
        "Gazprom": {"yesterday": 300.0, "today": 305.0}
    }
}
result = compute_spot_delta(example_input)
print(result)  # {'spot_deltas': {...}, 'total_spot_delta': 100.0, ... 'residual': 0.0}

# С реальными дельтами
real_deltas = {"Sber": 5.0, "VTB": -3.0, "Gazprom": 4.0}
result_with_real = compute_spot_delta(example_input, real_deltas=real_deltas)
print(result_with_real)
```

Сумма значений = 50.0 (ΔPV).

# explain_pl_by_elements
