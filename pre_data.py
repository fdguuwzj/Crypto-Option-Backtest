# -*- coding:utf-8 -*-
"""
@FileName：pre_data.py
@Description：
@Author：fdguuwzj
@Time：2024-10-10 11:40
"""
import pandas as pd
# 遍历每个唯一的snapshot_time
from joblib import Parallel, delayed
is_debug = False

def format_date(date_str):
    # 将 '27DEC24' 转换为 '2024-12-27'
    return pd.to_datetime(date_str, format='%d%b%y').strftime('%Y-%m-%d')


# data = pd.read_parquet('btc_option_data_toshare.parquet')
# data['mature_time'] = data['instrument_name'].str.split('-').str[1]
# data['exe_price'] = data['instrument_name'].str.split('-').str[2]
# data['type'] = data['instrument_name'].str.split('-').str[3]
# data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
# # 按snapshot_time排序
# data = data.sort_values('snapshot_time')
data = pd.read_pickle('data/btc_option_data.pkl')
# 删除bid或ask为NaN的行
data = data.dropna(subset=['bid_price', 'ask_price'])
# 将日期转换为 datetime 格式
data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
data['mature_time'] = pd.to_datetime(data['mature_time'])
data['mature_time'] = data['mature_time'].apply(format_date)
# debug
if is_debug:
    data = data.head(5000)
# 创建一个空的DataFrame来存储结果
time_spread = pd.DataFrame(columns=['snapshot_time'])
time_spread_list = []

fraction = 0.001
def sell(item, fraction):
    return item['bid_price']*(1-fraction)
def buy(item, fraction):
    return item['ask_price']*(1+fraction)

def process_time_group(time, time_data):
    max_spread = float('-inf')
    min_spread = float('inf')
    max_pair = min_pair = None
    min_mature = time_data['mature_time'].min()
    # 遍历每个唯一的到期时间和执行价格组合
    for (mature_exeprice, options) in time_data.groupby(['mature_time', 'exe_price']):
        if len(options) == 2 and mature_exeprice[0] == min_mature:  # 确保有看涨和看跌期权, 只选择近月期权
            call_option = options[options['type'] == 'C'].iloc[0]
            put_option = options[options['type'] == 'P'].iloc[0]
            call_price = call_option[['bid_price', 'ask_price']]
            call_option_name = call_option['instrument_name']
            put_price = put_option[['bid_price', 'ask_price']]
            put_option_name = put_option['instrument_name']
            up_spread = sell(call_price, fraction) - buy(put_price, fraction)
            down_spread = buy(call_price, fraction) - sell(put_price, fraction)
            if up_spread > max_spread:
                max_spread = up_spread
                max_pair = (mature_exeprice, call_option, put_option)
            if down_spread < min_spread:
                min_spread = down_spread
                min_pair = (mature_exeprice, call_option, put_option)

    # 将结果添加到time_spread DataFrame
    if max_pair and min_pair:
        max_mature_time, max_call_option, max_put_option = max_pair
        min_mature_time, min_call_option, min_put_option = min_pair
        return pd.DataFrame({
            'snapshot_time': time,
            'maxpair_mature_time': max_mature_time[0],
            'maxpair_exe_price': max_mature_time[1],
            'maxpair_spread': max_spread,
            'maxpair_call_price': max_call_option['mark_price'],
            'maxpair_put_price': max_put_option['mark_price'],
            'maxpair_call_option_name': max_call_option['instrument_name'],
            'maxpair_put_option_name': max_put_option['instrument_name'],
            'minpair_mature_time': min_mature_time[0],
            'minpair_exe_price': min_mature_time[1],
            'minpair_spread': min_spread,
            'minpair_call_price': min_call_option['mark_price'],
            'minpair_put_price': min_put_option['mark_price'],
            'minpair_call_option_name': min_call_option['instrument_name'],
            'minpair_put_option_name': min_put_option['instrument_name'],
        }, index=[0])
    return None

time_spread_list = Parallel(n_jobs=-1)(delayed(process_time_group)(time, time_data) for time, time_data in data.groupby('snapshot_time'))
time_spread = pd.concat(time_spread_list).reset_index(drop=True)


# for time, time_data in data.groupby('snapshot_time'):
#     max_spread = float('-inf')
#     min_spread = float('inf')
#     max_pair = min_pair = None
#
#     # 遍历每个唯一的到期时间和执行价格组合
#     for (mature_exeprice, options) in time_data.groupby(['mature_time', 'exe_price']):
#         if len(options) == 2:  # 确保有看涨和看跌期权
#             call_option = options[options['type'] == 'C'].iloc[0]
#             put_option = options[options['type'] == 'P'].iloc[0]
#             call_price = call_option['mark_price']
#             call_option_name = call_option['instrument_name']
#             put_price = put_option['mark_price']
#             put_option_name = put_option['instrument_name']
#             spread = call_price - put_price
#
#             if spread > max_spread:
#                 max_spread = spread
#                 max_pair = (mature_exeprice, call_option, put_option)
#             if spread < min_spread:
#                 min_spread = spread
#                 min_pair = (mature_exeprice, call_option, put_option)
#
#     # 将结果添加到time_spread DataFrame
#     if max_pair and min_pair:
#         max_mature_time, max_call_option, max_put_option = max_pair
#         min_mature_time, min_call_option, min_put_option = min_pair
#         time_spread_list.append(pd.DataFrame({
#             'snapshot_time': time,
#             'maxpair_mature_time': format_date(max_mature_time[0]),
#             'maxpair_exe_price': max_mature_time[1],
#             'maxpair_spread': max_spread,
#             'maxpair_call_price': max_call_option['mark_price'],
#             'maxpair_put_price': max_put_option['mark_price'],
#             'maxpair_call_option_name': max_call_option['instrument_name'],
#             'maxpair_put_option_name': max_put_option['instrument_name'],
#             'minpair_mature_time': format_date(min_mature_time[0]),
#             'minpair_exe_price': min_mature_time[1],
#             'minpair_spread': min_spread,
#             'minpair_call_price': min_call_option['mark_price'],
#             'minpair_put_price': min_put_option['mark_price'],
#             'minpair_call_option_name': min_call_option['instrument_name'],
#             'minpair_put_option_name': min_put_option['instrument_name'],
#         }, index=[0]))
# time_spread = pd.concat(time_spread_list).reset_index(drop=True)

print(time_spread)
time_spread.to_csv('min_max_options_couple.csv', encoding='utf-8', index=False)

# for time in data['snapshot_time'].unique():
    # time_data = data[data['snapshot_time'] == time]
    # max_spread = 0
    # min_spread = float('inf')
    # max_pair = min_pair = None

    # # 遍历每个唯一的到期时间和执行价格组合
    # for (mature_exeprice, options) in time_data.groupby(['mature_time', 'exe_price']):
    #     if len(options) == 2:  # 确保有看涨和看跌期权
    #         call_price = options[options['type'] == 'C']['mark_price'].values[0]
    #         call_option_name = options[options['type'] == 'C']['instrument_name'].values[0]
    #         put_price = options[options['type'] == 'P']['mark_price'].values[0]
    #         put_option_name = options[options['type'] == 'P']['instrument_name'].values[0]
    #         spread = call_price - put_price
    #         if spread > max_spread:
    #             max_spread = spread
    #             max_pair = (mature_exeprice, options)
    #             max_call_price = call_price
    #             max_put_price = put_price
    #             max_call_name = call_option_name
    #             max_put_name = put_option_name
    #         if spread < min_spread:
    #             min_spread = spread
    #             min_pair = (mature_exeprice, options)
    #             min_call_price = call_price
    #             min_put_price = put_price
    #             min_call_name = call_option_name
    #             min_put_name = put_option_name
    # # 将结果添加到time_spread DataFrame
    # if max_pair and min_pair:
    #     time_spread_list.append(pd.DataFrame({
    #         'snapshot_time': time,
    #         'maxpair_mature_time': format_date(max_pair[0][0]),
    #         'maxpair_exe_price': max_pair[0][1],
    #         'maxpair_spread': max_spread,  # 买一张put，卖一张call的收益
    #         'maxpair_call_price': max_call_price,
    #         'maxpair_put_price': max_put_price,
    #         'maxpair_call_option_name': max_call_name,
    #         'maxpair_put_option_name': max_put_name,
    #         'minpair_mature_time': format_date(min_pair[0][0]),
    #         'minpair_exe_price': min_pair[0][1],
    #         'minpair_spread': min_spread,  # 买一张put，卖一张call的收益
    #         'minpair_call_price': min_call_price,
    #         'minpair_put_price': min_put_price,
    #         'minpair_call_option_name': min_call_name,
    #         'minpair_put_option_name': min_put_name,
    #     }, index=[0]))
# time_spread = pd.concat(time_spread_list).reset_index(drop=True)
# 打印结果