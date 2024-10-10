# -*- coding:utf-8 -*-
"""
@FileName：main.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import pandas as pd
from loguru import logger
time_spread = pd.read_csv('data/min_max_options_couple.csv')
time_spread['minpair_mature_time'] = pd.to_datetime(time_spread['minpair_mature_time'])
time_spread['maxpair_mature_time'] = pd.to_datetime(time_spread['maxpair_mature_time'])
time_spread['snapshot_time'] = pd.to_datetime(time_spread['snapshot_time'])







class BackTrader:
    def __init__(self, current_position = {}, trade_positions = [], trade_profits = [], initial_capital = 100, time_spread=time_spread):
        # 创建一个字典来存储当前持仓
        self.current_position = current_position
        # 记录所有的持仓信息
        self.trade_positions = trade_positions
        # 创建一个列表来存储每次交易的收益
        self.trade_profits = trade_profits
        # 初始资金
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.time_spread = time_spread

    
    def arraive_time(self, current_time: pd.Timestamp):
        # 如果当前时间大于持仓的到期时间，平仓
        if current_time >= self.current_position['mature_time'] + pd.Timedelta(hours=8):
            return True
        else:
            return False


    def close_position(self, row):
        # 获取当前时间点的最大spread组合
        current_spread = row['maxpair_spread'] - row['minpair_spread']
        current_max_mature_time = row['maxpair_mature_time']
        current_max_exe_price = row['maxpair_exe_price']
        current_max_call_price = row['maxpair_call_price']
        current_max_call_name = row['maxpair_call_option_name']
        current_max_put_price = row['maxpair_put_price']
        current_max_put_name = row['maxpair_put_option_name']
        current_min_call_price = row['minpair_call_price']
        current_min_call_name = row['minpair_call_option_name']
        current_min_put_price = row['minpair_put_price']
        current_min_put_name = row['minpair_put_option_name']
        current_min_mature_time = row['minpair_mature_time']
        current_min_exe_price = row['minpair_exe_price']
        current_time = row['snapshot_time']

        self.current_position = {
            'max_call_price': current_max_call_price,
            'max_call_name': current_max_call_name,
            'max_put_price': current_max_put_price,
            'max_put_name': current_max_put_name,
            'min_call_price': current_min_call_price,
            'min_call_name': current_min_call_name,
            'min_put_price': current_min_put_price,
            'min_put_name': current_min_put_name,
            'current_time': current_time,
            'mature_time': max(current_min_mature_time, current_max_mature_time)
        }

        self.current_capital += current_spread
        logger.info(f"self.current_capital {self.current_capital}, current_spread {current_spread}")
        self.trade_profits.append({'current_time': current_time, 'current_spread': current_spread})
        self.trade_positions.append(self.current_position)


    # 创建一个函数来模拟交易
    def paper_trade(self, row):
        # 如果没有持仓，直接开仓
        if not self.current_position:
            self.close_position(row)
            # logger.info(f"开仓: {self.current_position}")
        else:
            if self.arraive_time(row['snapshot_time']):
                # 平仓
                self.close_position(row)
                # logger.info(f"平掉到期仓位后开仓: {self.current_position}")

    def trade(self):
        # 对time_spread中的每一行应用paper_trade函数
        self.time_spread.apply(self.paper_trade, axis=1)

        # 如果最后还有持仓，平掉它
        # if current_position:
        #     final_value = time_spread.iloc[-1]['max_spread'] - (current_position['call'] - current_position['put'])
        #     current_capital += final_value
        #     trade_profits.append(final_value)
        #     print(f"最终平仓: {current_position}")
        #     print(f"最终收益: {final_value}")

        # 计算总收益
        total_profit = self.current_capital - self.initial_capital
        total_return = (self.current_capital / self.initial_capital - 1) * 100

        print(f"初始资金: {self.initial_capital}")
        print(f"最终资金: {self.current_capital}")
        print(f"总收益: {total_profit}")
        print(f"总收益率: {total_return:.2f}%")
        trade_profits = pd.DataFrame(self.trade_profits)
        trade_positions = pd.DataFrame(self.trade_positions)
        trade_trails = pd.merge(trade_profits, trade_positions, on='current_time')
        # 计算其他统计数据
        num_trades = len(trade_profits)
        avg_profit_per_trade = trade_profits['current_spread'].sum() / num_trades if num_trades > 0 else 0
        max_profit = max(trade_profits['current_spread'])
        min_profit = min(trade_profits['current_spread'])

        print(f"交易次数: {num_trades}")
        print(f"平均每笔交易收益: {avg_profit_per_trade:.2f}")
        print(f"最大单笔收益: {max_profit:.2f}")
        print(f"最小单笔收益: {min_profit:.2f}")

if __name__ == '__main__':
    backtrader = BackTrader()
    backtrader.trade()
