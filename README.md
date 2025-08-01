# Explain PL by Elements: Расчёт разложения изменения PV по spot-факторам

## Описание
Этот проект (файл `pl_job.py`) реализует расчёт present value (PV) деривативного контракта и разложение изменения PV по компонентам (spot_delta, spot_cross_gamma, spot_residual) с использованием конечных разностей. Основан на QuantLib и аппроксимации Тейлора для нескольких базовых активов. Все расчёты в относительных единицах (% от номинала), без фактических сумм (notional).

Проект динамичен: поддерживает любое количество активов. Данные загружаются из файлов (market data, fixings, term).

## Зависимости
- Python 3.x
- Pandas, NumPy, SciPy, QuantLib
- Кастомные модули: common2, ql2 (с tools, QLContext, etc.)

## Шаги расчётов
Скрипт выполняет jobs в словаре `jobs` (1=вкл, 0=выкл). Каждый job использует QLCalculator для PV.

### 1. compute_pv_at_t0
- Загружает market data и fixings на T0 (valDate).
- Вычисляет базовый PV на T0 (global_t0_PV) в %.
- Сохраняет в calc_resalts.

### 2. compute_pv_with_t1_prices
- Загружает market data на T0, но fixings (цены активов) на T+1.
- valDate = T0 + 1 день.
- Вычисляет PV_t1 (PV на T+1 с ценами T+1, но market data T0).
- Используется для delta_pv = PV_t1 - global_t0_PV.

### 3. compute_spot_factors
- Основной job для spot_delta, gamma, cross_gamma, residual.
- Использует конечные разности с h = 0.0001 (0.01% сдвиг).
- Активы: new_assets_inf_fc (с учётом синтетических замен).

#### a. Spot Delta по активам
- Для каждого актива: PV_h = PV со сдвигом цены +h%.
- delta_pv = (PV_h - PV_base) / h  (%PV / %shift).
- delta_percent = (цена_T+1 - цена_T0) / цена_T0  (относительное изменение).
- spot_delta = delta_pv * delta_percent  (вклад в %PV).
- Сумма: total_spot_delta = ∑ spot_delta по активам.

#### b. Диагональная Gamma (по одному активу)
- Для каждого актива: PV_2h = PV со сдвигом +2h%.
- gamma = (PV_2h - 2*PV_h + PV_base) / (h²)  (%PV / (%shift)²).
- gamma_contrib = (gamma * (delta_percent)²) / 2  (вклад в %PV, с 1/2 для Тейлора).
- Сумма: diagonal_terms_sum = ∑ gamma_contrib.

#### c. Cross-Gamma (по парам активов)
- Пары: все уникальные (i < j).
- PV1 = PV +h% для asset1; PV2 = +h% для asset2; PV12 = +h% для обоих.
- cross_gamma = (PV12 - PV1 - PV2 + PV_base) / (h²)  (%PV / (%shift1 * %shift2)).
- cross_gamma_contrib = cross_gamma * (delta_percent1 * delta_percent2)  (вклад в %PV, без 1/2 — для mixed derivative).
- Сумма: cross_terms_sum = ∑ cross_gamma_contrib по парам.

#### d. Total Spot Cross Gamma и Residual
- total_spot_cross_gamma = cross_terms_sum + diagonal_terms_sum  (квадратичная часть в %PV).
- delta_pv = PV_t1 - PV_base  (полное изменение PV в %).
- spot_residual = delta_pv - total_spot_delta - total_spot_cross_gamma  (остаток — высшие порядки).

Все значения сохраняются в calc_resalts и экспортируются в Excel.

## Размерности
- PV: в % от номинала (0.9 = 90%).
- Сдвиги (h, delta_percent): в долях (0.01 = 1%).
- Delta/Gamma/Contrib/Residual: в %PV (относительное изменение).
- Для абсолютных сумм: умножьте на (notional / 100).

## Пример использования
Запустите `python pl_job.py`. Результаты в calc_resalts и Excel (укажите путь в df.to_excel).

- Вход: Файлы market data, fixings (T0/T+1), term (ZIP с trade info).
- Выход: Словарь/DF с PV, spot_delta по активам, gamma/cross_gamma по парам, sums, residual.

## Замечания
- Если job 'compute_pv_with_t1_prices' не запущен — residual не вычисляется (предупреждение).
- Для точности: h=0.0001; используйте симметричные разности для лучшей аппроксимации, если нужно.
- Документация: См. docs_tz/Методология...docx для методологии. 