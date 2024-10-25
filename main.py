# -*- coding:utf-8 -*-
"""
@FileName：main.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import pandas as pd

from strategies import BackTrader

if __name__ == '__main__':
    # backtrader = BoxSpreadBackTrader()
    # backtrader.trade()
    # backtrader.analyze_trade()
    # 双卖回测
    data = pd.read_pickle('data/btc_option_data_for_trade.pkl')
    backtrader = BackTrader(initial_capital=60000,data=data, date_interval=['2024-01-23 00:00:00', '2024-07-27 00:00:00'], fraction=0.01, exe_price_gear=1, mature_gear=0)
    backtrader.trade()
    backtrader.analyze_trade()
