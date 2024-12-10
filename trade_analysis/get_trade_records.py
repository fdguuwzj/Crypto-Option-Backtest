# -*- coding:utf-8 -*-
"""
@FileName：get_trade_records.py
@Description：
@Author：fdguuwzj
@Time：2024/12/3 17:11
"""
import os
import pandas as pd
import requests
import time
import hmac
import hashlib

from trade_analysis.config import bn_api_secret, bn_api_key

data_path = os.path.join('..', 'data', 'trades_data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

def create_signature(params):
    query_string = '&'.join([f"{key}={value}" for key, value in sorted(params.items())])
    return hmac.new(bn_api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def request_binance_api(endpoint, params):
    base_url = 'https://eapi.binance.com/eapi/v1'
    params['timestamp'] = int(time.time() * 1000)
    params['signature'] = create_signature(params)
    headers = {
        'X-MBX-APIKEY': bn_api_key
    }

    response = requests.get(f"{base_url}/{endpoint.strip()}", params=params, headers=headers)  # 去掉endpoint末尾的空格
    response.raise_for_status()  # 确保请求成功
    return response.json()

if __name__ == '__main__':
    symbols = ['BTC', 'ETH', 'DOGE']
    
    trades = request_binance_api('userTrades', {})  # 去掉endpoint末尾的空格
    trades = pd.DataFrame(trades)
    trades['time'] = pd.to_datetime(trades['time'], unit='ms')

    exercise = request_binance_api('exerciseRecord', {})
    exercise = pd.DataFrame(exercise)
    exercise['createDate'] = pd.to_datetime(exercise['createDate'], unit='ms')
    exercise.to_csv(f'{data_path}/exercise.csv', index=False)

    position = request_binance_api('position', {})
    if position:
        position = pd.DataFrame(position)
        position['time'] = pd.to_datetime(position['time'], unit='ms')
        position['expiryDate'] = pd.to_datetime(position['expiryDate'], unit='ms')
        position.to_csv(f'{data_path}/position.csv', index=False)
    else:
        print(f'no position currently.')
    trades_columns = ['time', 'symbol', 'price', 'quantity', 'fee', 'realizedProfit', 'side', 'optionSide', 'volatility']
    trades = trades[trades_columns]
    for symbol in symbols:
        symbol_trades = trades[trades['symbol'].str.contains(symbol)]
        symbol_trades.to_csv(f'{data_path}/{symbol}_trades.csv', index=False)