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

fraction = 0.001
def format_date(date_str):
    # 将 '27DEC24' 转换为 '2024-12-27'
    return pd.to_datetime(date_str, format='%d%b%y').strftime('%Y-%m-%d')


# data = pd.read_parquet('btc_option_data_toshare.parquet')
# data['mature_time'] = data['instrument_name'].str.split('-').str[1]
# data['exe_price'] = data['instrument_name'].str.split('-').str[2]
# data['type'] = data['instrument_name'].str.split('-').str[3]
# data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
# # 按snapshot_time排序
# # data = data.sort_values('snapshot_time')
# data = pd.read_pickle('data/btc_option_data.pkl')
# # 删除bid或ask为NaN的行
# data = data.dropna(subset=['bid_price', 'ask_price'])
# # 将日期转换为 datetime 格式
# data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
# data['mature_time'] = pd.to_datetime(data['mature_time'])
# data['mature_time'] = data['mature_time'].apply(format_date)


def sell(item, fraction = 0.001):
    # 进行盘口挂单
    # return item['bid_price']*(1-fraction)
    return item['mark_price']*(1+fraction)
def buy(item, fraction = 0.001):
    # 进行盘口挂单
    # return item['ask_price']*(1+fraction)
    return item['mark_price']*(1+fraction)
#
def process_time_group(time, time_data):
    min_mature = time_data['mature_time'].min()
    btc_price = time_data.loc[time_data['mature_time'] == min_mature, 'target_price'].mean()
    # 只在min_mature时遍历组合
    mature_data = time_data[time_data['mature_time'] == min_mature]
    long_box_max_pnl = float('-inf')
    long_box_best_H = long_box_best_L = None
    long_box_best_Ch = long_box_best_Ph = long_box_best_Cl = long_box_best_Pl = None
    long_box_best_H_option = long_box_best_L_option = None
    short_box_max_pnl = float('-inf')
    short_box_best_H = short_box_best_L = None
    short_box_best_Ch = short_box_best_Ph = short_box_best_Cl = short_box_best_Pl = None
    short_box_best_H_option = short_box_best_L_option = None
    # 遍历每个唯一的执行价格
    for H, H_options in mature_data.groupby('exe_price'):
        if len(H_options) == 2:  # 确保有看涨和看跌期权
            H_call_option = H_options[H_options['type'] == 'C'].iloc[0]
            H_put_option = H_options[H_options['type'] == 'P'].iloc[0]
            Ch = H_call_option[['bid_price', 'ask_price', 'mark_price']]
            Ph = H_put_option[['bid_price', 'ask_price', 'mark_price']]

            for L, L_options in mature_data.groupby('exe_price'):
                if len(L_options) == 2 and L < H:  # 确保有看涨和看跌期权，并且L < H
                    L_call_option = L_options[L_options['type'] == 'C'].iloc[0]
                    L_put_option = L_options[L_options['type'] == 'P'].iloc[0]
                    Cl = L_call_option[['bid_price', 'ask_price', 'mark_price']]
                    Pl = L_put_option[['bid_price', 'ask_price', 'mark_price']]

                    long_box_pnl = float(H) - float(L) + (- buy(Cl) + sell(Ch) + sell(Pl) - buy(Ph)) * btc_price
                    short_box_pnl = float(L) - float(H) + (sell(Cl) - buy(Ch) + buy(Pl) - sell(Ph)) * btc_price
                    if long_box_pnl > long_box_max_pnl:
                        long_box_max_pnl = long_box_pnl
                        long_box_best_H = H
                        long_box_best_L = L
                        long_box_best_sell_Ch = sell(Ch)
                        long_box_best_buy_Ph = buy(Ph)
                        long_box_best_buy_Cl = buy(Cl)
                        long_box_best_sell_Pl = sell(Pl)
                        long_box_best_H_option = (H_call_option, H_put_option)
                        long_box_best_L_option = (L_call_option, L_put_option)
                        long_box_best_mature = min_mature
                    if short_box_pnl > short_box_max_pnl:
                        short_box_max_pnl = short_box_pnl
                        short_box_best_H = H
                        short_box_best_L = L
                        short_box_best_buy_Ch = buy(Ch)
                        short_box_best_sell_Ph = sell(Ph)
                        short_box_best_sell_Cl = sell(Cl)
                        short_box_best_buy_Pl = buy(Pl)
                        short_box_best_H_option = (H_call_option, H_put_option)
                        short_box_best_L_option = (L_call_option, L_put_option)
                        short_box_best_mature = min_mature

    # 将结果添加到time_spread DataFrame
    if long_box_best_H is not None and long_box_best_L is not None and short_box_best_H is not None and short_box_best_L is not None:
        H_call_option, H_put_option = long_box_best_H_option
        L_call_option, L_put_option = long_box_best_L_option
        return pd.DataFrame({
            'snapshot_time': time,
            'long_box_max_pnl': long_box_max_pnl,
            'long_box_H': long_box_best_H,
            'long_box_L': long_box_best_L,
            'long_box_sell_Ch': long_box_best_sell_Ch,
            'long_box_buy_Ph': long_box_best_buy_Ph,
            'long_box_buy_Cl': long_box_best_buy_Cl,
            'long_box_sell_Pl': long_box_best_sell_Pl,
            'long_box_H_call_option_name': H_call_option['instrument_name'],
            'long_box_H_put_option_name': H_put_option['instrument_name'],
            'long_box_L_call_option_name': L_call_option['instrument_name'],
            'long_box_L_put_option_name': L_put_option['instrument_name'],
            'long_box_mature': long_box_best_mature,
            'short_box_max_pnl': short_box_max_pnl,
            'short_box_H': short_box_best_H,
            'short_box_L': short_box_best_L,
            'short_box_buy_Ch': short_box_best_buy_Ch,
            'short_box_sell_Ph': short_box_best_sell_Ph,
            'short_box_sell_Cl': short_box_best_sell_Cl,
            'short_box_buy_Pl': short_box_best_buy_Pl,
            'short_box_H_call_option_name': short_box_best_H_option[0]['instrument_name'],
            'short_box_H_put_option_name': short_box_best_H_option[1]['instrument_name'],
            'short_box_L_call_option_name': short_box_best_L_option[0]['instrument_name'],
            'short_box_L_put_option_name': short_box_best_L_option[1]['instrument_name'],
            'short_box_mature': short_box_best_mature,
            'target_price': btc_price
        }, index=[0])
    return None

if __name__ == '__main__':
    # debug
    # data = pd.read_parquet('data/btc_option_data_toshare.parquet')
    # data['mature_time'] = data['instrument_name'].str.split('-').str[1]
    # data['exe_price'] = data['instrument_name'].str.split('-').str[2]
    # data['type'] = data['instrument_name'].str.split('-').str[3]
    # data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
    # # 按snapshot_time排序
    # data = data.sort_values('snapshot_time')
    # data = pd.read_pickle('data/btc_option_data.pkl')
    # # 删除bid或ask为NaN的行
    # data = data.dropna(subset=['bid_price', 'ask_price'])
    # # 将日期转换为 datetime 格式
    # data['snapshot_time'] = pd.to_datetime(data['snapshot_time'])
    # data['mature_time'] = pd.to_datetime(data['mature_time'])
    # data['mature_time'] = data['mature_time'].apply(format_date)
    # # # data = pd.read_pickle('data/btc_option_data_for_trade.pkl')
    # data['snapshot_time'] = data['snapshot_time'].dt.floor('T')
    # # # # 创建一个空的DataFrame来存储结果
    # target_price = pd.read_feather('data/BTC-USDT.pkl')
    # target_price['candle_begin_time'] = pd.to_datetime(target_price['candle_begin_time'])
    # target_price.rename(columns={"candle_begin_time": 'snapshot_time'}, inplace=True)
    # target_price['target_price'] = target_price['open']
    # data = pd.merge(data, target_price[['snapshot_time', 'target_price']], on ='snapshot_time')
    # data.to_pickle('data/btc_option_data_for_trade.pkl')
    data = pd.read_pickle('data/btc_option_data_for_trade.pkl')
    # # if is_debug:
    # #     data = data.head(5000)
    # #
    print(data)
    time_spread_list = Parallel(n_jobs=-1)(delayed(process_time_group)(time, time_data) for time, time_data in data.groupby('snapshot_time'))
    time_spread = pd.concat(time_spread_list).reset_index(drop=True)
    print(time_spread)
    time_spread.to_csv('data/min_max_options_couple.csv', encoding='utf-8', index=False)

