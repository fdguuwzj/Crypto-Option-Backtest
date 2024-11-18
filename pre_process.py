
# -*- coding:utf-8 -*-
"""
@FileName：pre_process.py
@Description：
@Author：fdguuwzj
@Time：2024/11/11 11:29
"""
import pandas as pd

deribit_data = pd.read_pickle(r'data/all_processed_deribit_data1118.pkl')
# 双卖回测
data = pd.read_pickle('data/btc_option_data_for_trade1115.pkl')
deribit_data = deribit_data[['hour', 'expiration', 'strike_price', 'symbol', 'mark_price', 'type']]
deribit_data.rename(columns={'hour': 'snapshot_time',
                             'expiration': 'expiry_date',
                             'strike_price': 'exe_price',
                             'symbol': 'instrument_name',}, inplace=True)

deribit_data['type'] = deribit_data['type'].map({'call': 'C', 'put': 'P'})
deribit_data['bid_price'] = deribit_data['ask_price'] =  deribit_data['mark_price']
btc_data = pd.read_feather(r'data/BTC-USDT.pkl')
btc_data = btc_data[['candle_begin_time', 'open']]
btc_data.rename(columns={'open': 'btc_price'}, inplace=True)


data = data[['snapshot_time', 'expiry_date', 'exe_price', 'instrument_name', 'mark_price', 'bid_price', 'ask_price', 'type']]
final_data = pd.concat([deribit_data,data])
final_data = pd.merge(final_data, btc_data, left_on='snapshot_time', right_on='candle_begin_time', how='left')
del final_data['candle_begin_time']
final_data.sort_values(by='snapshot_time', inplace=True)

final_data.reset_index(inplace=True, drop=True)
# 找出重复数据
duplicates = final_data[final_data.duplicated(subset=['snapshot_time', 'instrument_name'], keep=False)]
print('重复数据数量:', len(duplicates))
print('\n重复数据示例:')
print(duplicates.sort_values(['snapshot_time', 'instrument_name']).head())
# 删除重复数据,保留第一条记录
final_data.drop_duplicates(subset=['snapshot_time', 'instrument_name'], keep='first', inplace=True)
print('删除重复数据后的数据量:', len(final_data))

final_data.to_pickle(r'data/btc_option_data_for_trade1118.pkl')
print(final_data)