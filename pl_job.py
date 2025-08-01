import logging, os, zipfile
from collections import namedtuple

import sys

import pandas as pd

sys.path.append("PATH”)


from common2 import tools as ctools
import ql2
from ql2 import tools, qlContext, consts as qlConsts
from ql2.products import _eqBasket



MARKET_DATA_FILE_NAME_ZERO = r PATH "


FIXINGS_FILE_NAME_ZERO = r" PATH "
FIXINGS_FILE_NAME_T_PLUS = r PATH "

TERM_FILE_NAME = r" PATH "



Named = namedtuple('Named', ('name', ))

log = logging.getLogger(__name__)

class _Trd(object):
    def __init__(self, posId, tradeId, productType, notional, qty, ccy, issueDate, maturityDate):
        self.id = posId
        self.tradeId = tradeId
        self.productType = Named(productType)
        self.notional = notional
        self.quantity = qty
        self.ccy = ccy
        self.issueDate = issueDate
        self.maturityDate = maturityDate
        self.modifiedAt = None

    def getKey(self, name):
        if name != 'ISIN': raise Exception('Misuse!')
        return len(self.tradeId) == 12


def _buildTerm(info):
    if isinstance(info, str):
        info = info.split('\n')
    posId, tradeId, productType, notional, qty, ccy, issueDate, maturityDate = None, None, None, None, None, None, None, None
    uls = None
    syntheticDef = None
    params = None
    for i, row in enumerate(info):
        row = row.strip()
        if row == '': continue
        if list(set(row)) == ['-']:
            params = ctools.parseYaml('\n'.join(info[i + 1:]))
            break
        xx = row.split(' = ')
        if len(xx) >= 2:
            if xx[0] == 'PosID':
                posId = int(xx[1])
            elif xx[0] == 'TradeID':
                tradeId = xx[1]
            elif xx[0] == 'ProductType':
                productType = xx[1]
            elif xx[0] == 'Notional':
                notional = float(xx[1]) if xx[1] is not None else None
            elif xx[0] == 'Qty':
                qty = float(xx[1]) if xx[1] is not None else None
            elif xx[0] == 'Ccy':
                ccy = xx[1]
            elif xx[0] == 'IssueDate':
                issueDate = ctools.toDate(xx[1]) if xx[1] is not None else None
            elif xx[0] == 'MaturityDate':
                maturityDate = ctools.toDate(xx[1]) if xx[1] is not None else None
            elif xx[0] == 'ULs':
                uls = xx[1].split(', ')
                syntheticDef = [None] * len(uls)
            elif xx[0].startswith('Synthetic.'):
                ul = xx[0][10:]
                si = list()
                for oneSyn in xx[1].split('|'):
                    oneSyn = oneSyn.split('->')
                    if len(oneSyn) == 2:
                        si.append((ctools.toDate(oneSyn[0].strip()), oneSyn[1].strip()))
                if len(info) > 0:
                    idx = uls.index(ul)
                    syntheticDef[idx] = _eqBasket.SyntheticEqData(ul, si)

    if posId is None: raise Exception("No PosID")
    if tradeId is None: raise Exception("No TradeID")
    if productType is None: raise Exception("No ProductType")
    if notional is None: pass  # allowed for stocks
    if qty is None: pass  # allowed for hedge/notes
    if ccy is None: raise Exception("No Ccy")
    if issueDate is None: pass  # allowed for stocks
    if maturityDate is None: pass  # allowed for stocks
    trade = _Trd(posId, tradeId, productType, notional, qty, ccy, issueDate, maturityDate)
    basket = []
    if uls is not None:
        basket = list(Named(ul) for ul in uls)
    term = tools.QLTerm(trade, basket=basket, params=params)
    if uls is not None:  # add synthetic, just in case
        term.syntheticDef = syntheticDef
    return term

def cleanTradeId(tradeId):
    for ch in (',', ' ', '\\', '/', ':', '+', '=', ):
        tradeId = tradeId.replace(ch, '_')
    return tradeId


def loadTerm(tradeId):
    _tradeId = cleanTradeId(tradeId)
    if not os.path.exists(TERM_FILE_NAME):
        raise Exception("No %s" % TERM_FILE_NAME)
    zf = zipfile.ZipFile(TERM_FILE_NAME)

    if _tradeId not in zf.NameToInfo:
        raise Exception("Unknown TradeID %s" % tradeId)
    info = zf.open(_tradeId, 'r').read().decode()
    print("info=", info)
    term = _buildTerm(info)
    zf.close()
    return term

def rename_asset_synthetic(syntheticDef, other_assets):
    import re
    # Парсим пары активов
    mapping = {}
    pattern = re.compile(r"eq='([^']+)'.+?'([\w\-]+)'\)\]")
    for item in syntheticDef:
        if item:
            left = item.eq
            right = item.info[0][1]
            mapping[left] = right.replace('"', '')
            # m = pattern.search(item)
            # if m:
            #     left, right = m.group(1), m.group(2)
            #     mapping[left] = right

    print('Пары замены:', mapping)

    # Заменяем активы по найденному сопоставлению
    new_assets = [mapping.get(asset, asset) for asset in other_assets]
    return new_assets


def refactor_prices(fixings, asset, date_target, epsilon=1.0001):
    import copy

    fixings_copy = copy.deepcopy(fixings)
    delta = None

    for i, (date_get, price) in enumerate(fixings_copy[asset]):
        if date_get == date_target:
            new_price = price * epsilon
            delta = new_price - price
            fixings_copy[asset][i] = (date_get, new_price)
            break
    else:
        print(f'нет цены для даты  = {date_target}')
    print(fixings_copy[asset])
    return delta, fixings_copy

def refactor_prices_pair_assets(fixings, asset_1, asset_2, date_target, epsilon=1.0001):
    import copy

    fixings_copy = copy.deepcopy(fixings)
    delta = None

    #Asset 1
    for i, (date_get, price) in enumerate(fixings_copy[asset_1]):
        if date_get == date_target:
            new_price = price * epsilon
            delta_1 = new_price - price
            delta_1_price = (asset_1, date_get, delta_1, new_price, price)
            fixings_copy[asset_1][i] = (date_get, new_price)
            break
    else:
        print(f'нет цены для даты  = {date_target} BA= {asset_1}')

    #Asset 2
    for i, (date_get, price) in enumerate(fixings_copy[asset_2]):
        if date_get == date_target:
            new_price = price * epsilon
            delta_2 = new_price - price
            delta_2_price = (asset_2, date_get, delta_2, new_price, price)
            fixings_copy[asset_2][i] = (date_get, new_price)
            break
    else:
        print(f'нет цены для даты  = {date_target} BA= {asset_2}')

    return delta_1_price, delta_2_price, fixings_copy

def calc_delta_price_between_dates(fixings, asset, date_last, date_future):
    import copy
    prev_price = None
    future_price = None


    for i, (date_get, price) in enumerate(fixings[asset]):
        if date_get == date_last:
            prev_price = price
        if date_get == date_future:
            future_price = price
        if prev_price and future_price:
            break
    try:
        delta_in_val = prev_price - future_price
        delta_percent = (future_price - prev_price) / prev_price
        return delta_percent, delta_in_val
    except Exception as ex:
        print(f"error = {ex}")
        return None




def make_calc(valDate, mdName, mdData, fixings, job, comment=None, for_asset_calc=None):
    qlCtx = qlContext.QLContext(valDate, mdName, mdData, fixings)

    calcParams = qlConsts.getCalcParamsDefault(valDate)
    qlCtx.setCalcParams(calcParams)

    qlCalc = ql2.QLCalculator(context)
    qlCalc.prepare()
    # qlRes содержит эти поля output которые считаются в других модулях
    outputs = ['PV', 'PV_IFRS', 'PV_RF', 'Seeds', 'Tolerance', 'Total Flow', 'Default Probability',
               'Coupon Probability', 'Coupon Flow', 'AutoCall Probability', 'AutoCall Flow', 'Accrued Coupon']
    # outputs = ql2.ALL_OUTPUTS
    print('Preparation finished. Calculating ...')

    qlRes = qlCalc._oneCalc(qlCtx, term, outputs)

    print("# %s" % tradeId)
    print('-' * 20)
    for k, v in qlRes.items():
        print("\t%s = %s" % (k, v))
    qlRes["tradeId"] = tradeId
    qlRes["job"] = job
    qlRes['clean_PV_val'] = float(str(qlRes['PV']).split()[0])
    qlRes['clean_PV_cur'] = str(qlRes['PV']).split()[1]
    if comment:
        qlRes["comment"] = comment
    if for_asset_calc:
        qlRes["for_asset_calc"] = for_asset_calc
    print("СЛОВАРЬ:", qlRes)
    return qlRes

def get_assets_pairs(assets_list):
    pairs = []
    n = len(assets_list)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((assets_list[i], assets_list[j]))
    return pairs


if __name__ == "__main__":
    import datetime
    from datetime import timedelta
    import sys, numpy as np, scipy, QuantLib as ql
    # print("Python %s / NumPy %s / SciPy %s / QuantLib %s" % (sys.version, np.__version__, scipy.__version__, ql.__version__))
    import common2
    context = common2.getGlobalContext()



    tradeId = “id”

    term = loadTerm(tradeId)
    # Синтетические БА заменить надо для корректировки маркет даты
    new_assets_inf_fc = rename_asset_synthetic(term.syntheticDef, term.uls)

    jobs = {
        'calc_PV_date_zero': 1,
        'calc_PV_shift_t1_prices': 1,
        'calc_PV_shift_prices_0_01_per_t0': 1,
        'calc_pairs_shift_PV': 1
    }

    calc_resalts = []
    global_t0_PV = None

    for job, status in jobs.items():
        if status:
           if job == 'calc_PV_date_zero':
                valDate, mdName, mdData = qlContext._loadMarketData(MARKET_DATA_FILE_NAME_ZERO)
                fixings_real = qlContext._loadFixings(FIXINGS_FILE_NAME_ZERO)
                result = make_calc(
                    valDate=valDate,
                    mdName=mdName,
                    mdData=mdData,
                    fixings=fixings_real,
                    job=job,
                    comment=f'Делаем расчет на дату {valDate}, по ценам на {valDate}'
                )

                global_t0_PV = result['clean_PV_val']
                calc_resalts.append(result)

            if job == 'calc_PV_shift_t1_prices':
                valDate, mdName, mdData = qlContext._loadMarketData(MARKET_DATA_FILE_NAME_ZERO)
                fixings_real = qlContext._loadFixings(FIXINGS_FILE_NAME_T_PLUS)
                # Нужно сдвинуть дату для расчетов valDate берется из названия файла а оно Т0
                valDate = valDate + timedelta(days=1)
                result = make_calc(
                    valDate=valDate,
                    mdName=mdName,
                    mdData=mdData,
                    fixings=fixings_real,
                    job=job,
                    comment=f'Делаем расчет на дату {valDate}, по ценам на {valDate}, маркетдата на {valDate - timedelta(days=1)}'
                )
                calc_resalts.append(result)
            if job == 'calc_PV_shift_prices_0_01_per_t0':
                valDate, mdName, mdData = qlContext._loadMarketData(MARKET_DATA_FILE_NAME_ZERO)
                fixings_real = qlContext._loadFixings(FIXINGS_FILE_NAME_T_PLUS)
                epsilon_percent = 10.0001
                for asset in new_assets_inf_fc:
                    # Меняем маркет дату
                   date_target = datetime.datetime(2025, 3, 12).date()
                    print(f"valdate == targetdate = {valDate==date_target}")
                    delta, fixings = refactor_prices(fixings_real, asset, valDate, epsilon_percent)
                    result = make_calc(
                        valDate=valDate,
                        mdName=mdName,
                        mdData=mdData,
                        fixings=fixings,
                        job=job,
                        comment=f"PV_BA=[{asset}]_незначительный_рост_ба",
                        for_asset_calc=asset
                    )
                    result['old_price_last'] = fixings_real[asset][-2:]
                    result['new_price_last'] = fixings[asset][-2:]
                    result['delta_change_asset_percent'], result['delta_change_asset_in_val'] = calc_delta_price_between_dates(
                        fixings=fixings_real,
                        asset=asset,
                        date_last=valDate,
                        date_future = valDate + timedelta(days=1)
                    )
                    result['epsilon_in_points_notional'] = delta
                    result['epsilon_in_percent'] = epsilon_percent - 1
                    result['global_t0_PV'] = global_t0_PV
                    tmp_delta_pv = result['clean_PV_val'] - result['global_t0_PV']
                    if tmp_delta_pv != 0.0:
                        result['delta_PV'] = (result['clean_PV_val'] - result['global_t0_PV']) / result['epsilon_in_percent']
                    else:
                        result['delta_PV'] = 0
                    result['spot_delta'] = result['delta_PV'] * result['delta_change_asset_percent']
                    calc_resalts.append(result)
            if job == 'calc_pairs_shift_PV':
                valDate, mdName, mdData = qlContext._loadMarketData(MARKET_DATA_FILE_NAME_ZERO)
                fixings_real = qlContext._loadFixings(FIXINGS_FILE_NAME_T_PLUS)
                epsilon = 1.0001  # Малый сдвиг для finite difference, исправляем с 10.0001
                h = epsilon - 1  # Размер сдвига в относительных единицах

                # Базовый PV на T0 (используем global_t0_PV или рассчитываем заново если нужно
                # Но предполагаем, что global_t0_PV уже есть; если нет, добавить расчёт
                PV_base = global_t0_PV

                pairs_list = get_assets_pairs(new_assets_inf_fc)
                
                import multiprocessing
                from functools import partial

                def calc_pv_for_shift(fixings_base, shifts, valDate, mdName, mdData):
                    fixings = copy.deepcopy(fixings_base)
                    for asset, delta_mult in shifts.items():
                        for i, (date_get, price) in enumerate(fixings[asset]):
                            if date_get == valDate:
                                fixings[asset][i] = (date_get, price * (1 + delta_mult))
                                break
                    return make_calc(valDate, mdName, mdData, fixings, job, comment="Shift calc", for_asset_calc=asset)

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Параллелизация

                results = []
                for pair in pairs_list:
                    asset1, asset2 = pair
                    
                    # Индивидуальные сдвиги
                    shift1 = {asset1: h}
                    res1 = pool.apply_async(calc_pv_for_shift, (fixings_real, shift1, valDate, mdName, mdData))
                    
                    shift2 = {asset2: h}
                    res2 = pool.apply_async(calc_pv_for_shift, (fixings_real, shift2, valDate, mdName, mdData))
                    
                    # Двойной сдвиг
                    shift12 = {asset1: h, asset2: h}
                    res12 = pool.apply_async(calc_pv_for_shift, (fixings_real, shift12, valDate, mdName, mdData))
                    
                    # Получаем результаты
                    PV1 = res1.get()['clean_PV_val']
                    PV2 = res2.get()['clean_PV_val']
                    PV12 = res12.get()['clean_PV_val']
                    
                    # Расчёт cross_gamma
                    cross_gamma = (PV12 - PV1 - PV2 + PV_base) / (h * h)  # Поскольку h1 = h2 = h
                    
                    # Delta изменений цен (аналогично существующему)
                    delta_percent1, delta_val1 = calc_delta_price_between_dates(fixings_real, asset1, valDate, valDate + timedelta(days=1))
                    delta_percent2, delta_val2 = calc_delta_price_between_dates(fixings_real, asset2, valDate, valDate + timedelta(days=1))
                    
                    # Компонент вклада в spot_cross_gamma
                    cross_gamma_contrib = cross_gamma * (delta_val1 * delta_val2)  # Или delta_percent, в зависимости от единиц; по методологии delta_S1 * delta_S2
                    
                    # Сохраняем в result
                    result = {
                        'job': job,
                        'for_asset_calc': pair,
                        'cross_gamma': cross_gamma,
                        'cross_gamma_contrib': cross_gamma_contrib,
                        'PV_base': PV_base,
                        'PV1': PV1,
                        'PV2': PV2,
                        'PV12': PV12,
                        'delta_S1': delta_val1,
                        'delta_S2': delta_val2,
                        # ... добавить другие поля если нужно
                    }
                    # TODO: Для сокращения O(n^2) - фильтровать pairs_list по коррелированным активам, напр. если corr(asset1, asset2) > threshold
                    results.append(result)
                
                pool.close()
                pool.join()
                
                calc_resalts.extend(results)  # Вместо append по одному
    try:
        df= pd.DataFrame(calc_resalts)
        df.to_excel(r" PATH ")
    except Exception as ex:
        print(f"Error = {ex}")

 


Ответить