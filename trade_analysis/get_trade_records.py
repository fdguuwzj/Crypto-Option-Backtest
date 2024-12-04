# -*- coding:utf-8 -*-
"""
@FileName：get_trade_records.py
@Description：
@Author：fdguuwzj
@Time：2024/12/3 17:11
"""
import os

import pandas as pd

from utils.binance import get_binance_ex
from utils.utils import retry_wrapper

data_path = os.path.join('..', 'data', 'trades_data')
if not os.path.exists(data_path):
    os.mkdir(data_path)
# [_ for _ in dir(bn) if 'position' in _]
if __name__ == '__main__':
    symbols = ['BTC', 'ETH', 'DOGE']
    bn = get_binance_ex()
    trades = retry_wrapper(
        bn.eapiPrivateGetUserTrades, params={'timestamp': ''}, func_name='eapiPrivateGetUserTrades'
    )
    trades = pd.DataFrame(trades)
    trades['time'] = pd.to_datetime(trades['time'], unit='ms')

    exercise = retry_wrapper(
        bn.eapiprivate_get_exerciserecord, params={'timestamp': ''}, func_name='eapiprivate_get_exerciserecord'
    )

    exercise = pd.DataFrame(exercise)
    exercise['createDate'] = pd.to_datetime(exercise['createDate'], unit='ms')
    exercise.to_csv(f'{data_path}/exercise.csv', index=False)
    position = retry_wrapper(
        bn.eapiprivate_get_position, params={'timestamp': ''}, func_name='eapiprivate_get_position'
    )
    position = pd.DataFrame(position)
    position['time'] = pd.to_datetime(position['time'], unit='ms')
    position['expiryDate'] = pd.to_datetime(position['expiryDate'], unit='ms')
    position.to_csv(f'{data_path}/position.csv', index=False)
    trades_columns = ['time', 'symbol', 'price', 'quantity', 'fee', 'realizedProfit', 'side', 'optionSide', 'volatility']
    trades = trades[trades_columns]
    for symbol in symbols:
        symbol_trades = trades[trades['symbol'].str.contains(symbol)]
        symbol_trades.to_csv(f'{data_path}/{symbol}_trades.csv', index=False)
    # print(1)