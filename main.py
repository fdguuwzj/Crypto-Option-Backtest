# -*- coding:utf-8 -*-
"""
@FileName：main.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import pandas as pd
from loguru import logger
from plotly.subplots import make_subplots

time_spread = pd.read_csv('data/min_max_options_couple.csv')
time_spread['short_box_mature'] = pd.to_datetime(time_spread['short_box_mature'])
time_spread['long_box_mature'] = pd.to_datetime(time_spread['long_box_mature'])
time_spread['snapshot_time'] = pd.to_datetime(time_spread['snapshot_time'])







class BackTrader:
    def __init__(self, current_position = {}, trade_positions = [], trade_profits = [], initial_capital = 30000, time_spread=time_spread):
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
        self.num_trades = 0

    
    def arraive_time(self, current_time: pd.Timestamp):
        # 如果当前时间大于持仓的到期时间，平仓
        if 'mature_time' not in self.current_position.keys():
            return True
        elif current_time >= self.current_position['mature_time'] + pd.Timedelta(hours=8):
            return True
        else:
            return False


    def close_position(self, row):
        # 获取当前时间点的最大spread组合
        current_time = row['snapshot_time']
        current_long_box_max_pnl = row['long_box_max_pnl']
        current_short_box_max_pnl = row['short_box_max_pnl']
        if current_long_box_max_pnl < 0 and current_short_box_max_pnl < 0:
            self.current_position = {'current_time': current_time}
            logger.info("两个pnl都小于0，不开仓")
            self.trade_profits.append({'current_time': current_time, 'current_spread': 0})
            self.trade_positions.append(self.current_position)
            return

        elif current_long_box_max_pnl > current_short_box_max_pnl:
            current_max_call_price = row['long_box_sell_Ch']
            current_max_call_name = row['long_box_H_call_option_name']
            current_max_put_price = row['long_box_sell_Pl']
            current_max_put_name = row['long_box_L_put_option_name']
            current_min_call_price = row['long_box_buy_Cl']
            current_min_call_name = row['long_box_L_call_option_name']
            current_min_put_price = row['long_box_buy_Ph']
            current_min_put_name = row['long_box_H_put_option_name']
            current_spread = current_long_box_max_pnl
            current_min_mature_time = row['long_box_mature']
            current_max_mature_time = row['long_box_mature']
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
            self.trade_profits.append({'current_time': current_time, 'current_spread': current_spread})
            logger.info(f"self.current_capital {self.current_capital}, current_spread {current_spread}")
            self.trade_positions.append(self.current_position)
            self.num_trades += 1
        else:
            current_max_call_price = row['short_box_buy_Ch']
            current_max_call_name = row['short_box_H_call_option_name']
            current_max_put_price = row['short_box_buy_Pl']
            current_max_put_name = row['short_box_L_put_option_name']
            current_min_call_price = row['short_box_sell_Cl']
            current_min_call_name = row['short_box_L_call_option_name']
            current_min_put_price = row['short_box_sell_Ph']
            current_min_put_name = row['short_box_H_put_option_name']
            current_spread = current_short_box_max_pnl
            current_min_mature_time = row['short_box_mature']
            current_max_mature_time = row['short_box_mature']
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
            self.trade_profits.append({'current_time': current_time, 'current_spread': current_spread})
            logger.info(f"self.current_capital {self.current_capital}, current_spread {current_spread}")
            self.trade_positions.append(self.current_position)
            self.num_trades += 1


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
        print(f"年化收益率: {total_return**(24*365/len(self.time_spread)):.2f}%")
        trade_profits = pd.DataFrame(self.trade_profits)
        trade_positions = pd.DataFrame(self.trade_positions)
        self.trade_trails = pd.merge(trade_profits, trade_positions, on='current_time')
        # 计算其他统计数据

        avg_profit_per_trade = trade_profits['current_spread'].sum() / self.num_trades if self.num_trades > 0 else 0
        max_profit = max(trade_profits['current_spread'])
        min_profit = min(trade_profits['current_spread'])

        print(f"交易次数: {self.num_trades}")
        print(f"平均每笔交易收益: {avg_profit_per_trade:.2f}")
        print(f"最大单笔收益: {max_profit:.2f}")
        print(f"最小单笔收益: {min_profit:.2f}")


    def analyze_trade(self):
        import plotly.graph_objects as go
        import pandas as pd

        # 计算累积净值
        self.trade_trails['cumulative_return'] = (self.trade_trails['current_spread']).cumsum()

        # 计算最大回撤
        self.trade_trails['cumulative_max'] = self.trade_trails['cumulative_return'].cummax()
        self.trade_trails['drawdown'] = self.trade_trails['cumulative_return'] / self.trade_trails['cumulative_max'] - 1

        # 创建收益曲线图
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # 添加累积净值折线图
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['cumulative_return'],
                                 mode='lines+markers', name='累积净值'), row=1, col=1)
        # 调整每笔收益散点图以使其更加显眼，并在每个点上标明此单收益
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['current_spread'],
                                 mode='markers+text', name='每笔收益', marker_color='green', opacity=1, 
                                 text=[f"{round(value, 2)}" for value in self.trade_trails['current_spread'] ], textposition='top center',
                                 textfont=dict(size=14, color='black')), row=2, col=1)

        # 更新布局
        fig.update_layout(
            title='收益曲线图',
            xaxis_title='时间',
            yaxis_title='累积净值',
            yaxis2_title='单笔收益',
            legend=dict(x=0, y=1.2, orientation='h')
        )

        fig.show()

if __name__ == '__main__':
    backtrader = BackTrader()
    backtrader.trade()
    backtrader.analyze_trade()
