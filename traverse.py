# -*- coding:utf-8 -*-
"""
@FileName：get_trade_records.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import os
import pandas as pd
from config import BACKTEST_DIR, BACKTEST_SYMBOL_DIR, SNAPSHOT_TIME, BID_PRICE, MARK_PRICE, ASK_PRICE, EXPIRATION, \
    EXE_PRICE, TYPE, OPTION_NAME
from strategies import BackTrader, TradingLogger
from utils.options_utils import btc_vol_df
from utils.utils import time_it, timer
import multiprocessing

target = 'SOL'
with timer('read data'):
    # data = pd.read_pickle(os.path.join(BACKTEST_DIR, 'btc_option_data_for_trade_all_year.pkl'))
    # data = pd.read_pickle(os.path.join(BACKTEST_DIR, 'btc_option_data_for_trade1118.pkl'))
    data = pd.read_pickle(os.path.join(BACKTEST_SYMBOL_DIR, f'{target}_data_23_24_25.pkl'))
    data = data.rename(
        columns={"hour": SNAPSHOT_TIME, 'expiration': EXPIRATION, 'strike_price': EXE_PRICE, 'type': TYPE,
                 'mark_price': MARK_PRICE, 'symbol': OPTION_NAME,
                 'bid_price': BID_PRICE, 'ask_price': ASK_PRICE})
    data['bid_price'] = data['bid_price'].fillna(0)
    data['bid_iv'] = data['bid_iv'].fillna(0)
    data['ask_price'] = data['ask_price'].fillna(0)
    data['ask_iv'] = data['ask_iv'].fillna(0)
    print(data.head(10).to_markdown())

def run_backtrader(n):
    trading_logger = TradingLogger()
    backtrader = BackTrader(initial_capital=10000,
                            strategy_params={'name': 'sell_straddle', 'exe_price_gear': n, 'mature_gear': 3},
                            data=data, date_interval=['2024-01-01 00:00:00', '2024-12-31 00:00:00'],
                            fraction=0.01, open_type='num_value', open_value=50, quote_type='um',
                            trading_logger=trading_logger, target=target)
    backtrader.trade_with_ddh(hedge_type='use_target_hedge_channel')
    backtrader.analyze_trade2(info='use_target_hedge_channel', show=False)


if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run_backtrader, range(3, 8))







