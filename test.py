# -*- coding:utf-8 -*-
"""
@FileName：test.py
@Description：
@Author：fdguuwzj
@Time：2024-10-10 14:26
"""
import pandas as pd

# def format_date(date_str):
#     # 将 '27DEC24' 转换为 '2024-12-27'
#     return pd.to_datetime(date_str, format='%d%b%y').strftime('%Y-%m-%d')

# # data = data.sort_values('snapshot_time')
# data = pd.read_pickle('data/btc_option_data.pkl')
# # 删除bid或ask为NaN的行
# data = data.dropna(subset=['bid_price', 'ask_price'])
# # 将日期转换为 datetime 格式
# data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
# data['mature_time'] = pd.to_datetime(data['mature_time'])
# data['mature_time'] = data['mature_time'].apply(format_date)
# min_mature = data.groupby('snapshot_time')['mature_time'].min().reset_index()
# min_mature.to_csv('min_mature_time.csv', index=False)


import os
import shutil

# 源文件夹和目标文件夹路径
src_dir = 'path/to/source/folder'
dst_dir = 'path/to/destination/folder'
tmp = [
    ['XNO', '2025-02-07', '2025-02-11'],
    ['STPT', '2025-01-25', '2025-01-27'],
    ['COTI', '2024-11-22', '2024-11-23'],
    ['COTI', '2024-12-09', '2024-12-10'],
    ['IDEX', '2024-12-08', '2024-12-10'],
    ['IDEX', '2024-12-21', '2024-12-23'],
    ['IDEX', '2024-11-28', '2024-11-29'],
    ['OG', '2024-10-05', '2024-10-09'],
    ['LIT', '2025-02-01', '2025-02-04'],
    ['POND', '2024-11-17', '2024-11-19'],
    ['POND', '2024-11-26', '2024-11-27'],
    ['POND', '2024-12-09', '2024-12-11'],
    ['POND', '2024-12-27', '2024-12-28'],
    ['POND', '2025-01-22', '2025-01-24'],

       ]
# 遍历tmp列表
for item in tmp:
    symbol, start_date, end_date = item
    

    # 将日期字符串转换为datetime对象
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 生成日期范围内的所有日期
    date_range = pd.date_range(start, end)
    
    # 遍历日期范围内的每一天
    for date in date_range:
        # 格式化日期为字符串
        date_str = date.strftime('%Y-%m-%d')
        # 构造文件名
        filename = f"{date_str}_{symbol}.csv"
        # 源文件路径
        src_file = os.path.join(src_dir, filename)
        
        # 如果文件存在则复制
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_dir)
            print(f"已复制文件: {filename}")
        else:
            print(f"文件不存在: {filename}")
 
 