# -*- coding:utf-8 -*-
"""
@FileName：test.py
@Description：
@Author：fdguuwzj
@Time：2024-10-10 14:26
"""
import pandas as pd

def format_date(date_str):
    # 将 '27DEC24' 转换为 '2024-12-27'
    return pd.to_datetime(date_str, format='%d%b%y').strftime('%Y-%m-%d')

# data = data.sort_values('snapshot_time')
data = pd.read_pickle('data/btc_option_data.pkl')
# 删除bid或ask为NaN的行
data = data.dropna(subset=['bid_price', 'ask_price'])
# 将日期转换为 datetime 格式
data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
data['mature_time'] = pd.to_datetime(data['mature_time'])
data['mature_time'] = data['mature_time'].apply(format_date)
min_mature = data.groupby('snapshot_time')['mature_time'].min().reset_index()
min_mature.to_csv('min_mature_time.csv', index=False)