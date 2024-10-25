# -*- coding:utf-8 -*-
"""
@FileName：strategies.py
@Description：
@Author：fdguuwzj
@Time：2024/10/21 16:15
"""
from math import sqrt

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
import plotly.graph_objects as go
data = pd.read_csv('data/min_max_options_couple.csv')

class Option:
    def __init__(self, snapshot_time: str = '0',
                 name: str = '0', ask_price: float = 0,
                 bid_price: float = 0, mark_price: float = 0,
                 expiry_date: pd.Timestamp = 0, exe_price: float = 0.0,
                 type: str = 'C', btc_price: float = 0.0, side: str = 'buy'):
        self.snapshot_time = snapshot_time
        self.name = name
        self.ask_price = ask_price
        self.bid_price = bid_price
        self.mark_price = mark_price
        self.expiry_date = expiry_date
        self.exe_price = exe_price
        self.type = type
        self.btc_price = btc_price
        self.side = side # 'buy' or 'sell'


class Position:
    def __init__(self, current_time: pd.Timestamp, options: list[Option] = None):
        self.current_time = current_time
        self.options: list[Option] = options if options else None
        self.expiry_date = max([option.expiry_date for option in options]) if options else None

    


class BackTrader:
    def __init__(self, current_position: Position=None, trade_positions=[], trade_profits=[], initial_capital=30000,
                 data=data, fraction=0.001, exe_price_gear=1, mature_gear=0, date_interval = ['2024-01-01 00:00:00', '2024-07-27 00:00:00']):
        # 创建一个字典来存储当前持仓

        self.current_position = current_position
        # 记录所有的持仓信息
        self.trade_positions = trade_positions
        # 创建一个列表来存储每次交易的收益
        self.trade_profits = trade_profits
        # 交易次数
        self.num_trades = 0
        # 初始资金
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        # 输入数据
        self.data = data
        self.target_data =  data[['snapshot_time', 'btc_price']].drop_duplicates().set_index('snapshot_time')
        self.time_stamps = data[(data['snapshot_time'] >= date_interval[0]) & (data['snapshot_time'] <= date_interval[1])]['snapshot_time'].unique()
        # 输出data数据的起讫日期
        logger.info(f'data数据的起始日期: {self.data["snapshot_time"].min()}, 结束日期: {self.data["snapshot_time"].max()}')
        logger.info(f'回测起始日期: {max(pd.to_datetime(date_interval[0]), self.data["snapshot_time"].min())}, 回测结束日期: {min(pd.to_datetime(date_interval[1]), self.data["snapshot_time"].max())}, 总共时长: {(pd.to_datetime(date_interval[1]) - pd.to_datetime(date_interval[0])).total_seconds() / 3600} h')
        # 交易设置
        self.exe_price_gear = exe_price_gear
        self.mature_gear = mature_gear
        self.fraction = fraction


    def sell(self, item, amount=1, opponent=True):
        # 进行盘口挂单
        if opponent:
            return item.bid_price*(1-self.fraction)*amount
        else:
            return item.mark_price * (1 + self.fraction)*amount

    def buy(self, item, amount=1, opponent=True):
        # 进行盘口挂单
        if opponent:
            return item.ask_price*(1 + self.fraction)*amount
        else:
            return item.mark_price * (1 + self.fraction)*amount


    def arraive_time(self, current_time: pd.Timestamp):
        # 如果当前时间大于持仓的到期时间，平仓
        if  not hasattr(self.current_position, 'expiry_date'):
            return True
        elif current_time >= self.current_position.expiry_date + pd.Timedelta(hours=8):
            return True
        else:
            return False

    def close_position(self, current_time: pd.Timestamp):
        # 持仓 -》 空仓
        logger.info(f'{current_time}: close_position')
        now_btc_price = self.target_data.loc[current_time, 'btc_price']
        # 只考虑到期行权情况
        for option in self.current_position.options:
            if option.side == 'buy' and option.type == 'C':
                position_pnl = max(now_btc_price - option.exe_price, 0)
            elif option.side == 'sell' and option.type == 'C':
                position_pnl = min(-now_btc_price + option.exe_price, 0)
            elif option.side == 'buy' and option.type == 'P':
                position_pnl = max(-now_btc_price + option.exe_price , 0)
            elif option.side == 'sell' and option.type == 'P':
                position_pnl = min(now_btc_price - option.exe_price, 0)
        self.trade_profits.append({'current_time': current_time, 'pnl': position_pnl})
        self.current_capital += position_pnl
        self.current_position = Position(current_time)

    def open_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓 
        # 开近月平值双卖
        mature_dates = sorted(time_options['expiry_date'].unique())
        mature_date = self.get_mature_date(mature_dates)
        time_options = time_options[time_options['expiry_date'] == mature_date]
        btc_price = time_options['btc_price'].values[0]
        exe_prices = sorted(time_options['exe_price'].astype('int').unique())
        down_exe_price, up_exe_prices = self.get_exe_price(btc_price, exe_prices)
        down_put_row = time_options[(time_options['exe_price'].astype('int') == down_exe_price)& (time_options['type'] == 'P')]
        up_call_row = time_options[(time_options['exe_price'].astype('int') == up_exe_prices)& (time_options['type'] == 'C')]
        config = {
            'snapshot_time': down_put_row['snapshot_time'].values[0],
            'name': down_put_row['instrument_name'].values[0],
            'mark_price': down_put_row['mark_price'].values[0],
            'bid_price': down_put_row['bid_price'].values[0],
            'ask_price': down_put_row['ask_price'].values[0],
            'expiry_date': down_put_row['expiry_date'].values[0],
            'exe_price': float(down_put_row['exe_price'].values[0]),
            'type': down_put_row['type'].values[0],
            'btc_price': down_put_row['btc_price'].values[0],
                           }
        down_put = Option(snapshot_time=config['snapshot_time'], name=config['name'], mark_price=config['mark_price'],
                          bid_price=config['bid_price'], ask_price=config['ask_price'],exe_price=config['exe_price'],
                          expiry_date=config['expiry_date'], type=config['type'], btc_price=config['btc_price'], side='sell',)
        config['snapshot_time'] = up_call_row['snapshot_time'].values[0]
        config['name'] = up_call_row['instrument_name'].values[0]
        config['mark_price'] = up_call_row['mark_price'].values[0]
        config['bid_price'] = up_call_row['bid_price'].values[0]
        config['ask_price'] = up_call_row['ask_price'].values[0]
        config['expiry_date'] = up_call_row['expiry_date'].values[0]
        config['exe_price'] = float(up_call_row['exe_price'].values[0])
        config['type'] = up_call_row['type'].values[0]
        config['btc_price'] = up_call_row['btc_price'].values[0]
        up_call = Option(snapshot_time=config['snapshot_time'], name=config['name'], mark_price=config['mark_price'],
                          bid_price=config['bid_price'], ask_price=config['ask_price'],exe_price=config['exe_price'],
                          expiry_date=config['expiry_date'], type=config['type'], btc_price=config['btc_price'], side='sell')
        self.current_position = Position(current_time, [down_put, up_call])
        logger.info(f'{current_time}: position opened {self.current_position}')
        position_pnl =  self.sell(down_put)*down_put.btc_price + self.sell(up_call)*up_call.btc_price
        self.trade_profits.append({'current_time': current_time, 'pnl': position_pnl})
        self.current_capital += position_pnl
        self.trade_positions.append(vars(self.current_position))
        self.num_trades += 1


    def get_mature_date(self, mature_dates):
        # 开近月期权
        return mature_dates[self.mature_gear]

    def get_exe_price(self, btc_price: float, exe_prices: list):
        # 设置参数n，找出第n大的数，和第n小的数
        lower_exe_prices = sorted([exe_price for exe_price in exe_prices if exe_price < btc_price], reverse=True)[:self.exe_price_gear]
        upper_exe_prices = sorted([exe_price for exe_price in exe_prices if exe_price > btc_price], reverse=False)[:self.exe_price_gear]
        min_exe_price = lower_exe_prices[-1] if lower_exe_prices else None
        max_exe_price = upper_exe_prices[-1] if upper_exe_prices else None
        return min_exe_price, max_exe_price

    def trade(self):


        for time_stamp in self.time_stamps:
            current_time = pd.to_datetime(time_stamp)
            time_data = self.data[self.data['snapshot_time'] == current_time]
            if self.current_position:
                if self.arraive_time(time_stamp):
                    # close position
                    self.close_position(time_stamp)
                    # open new position
                    self.open_position(time_stamp, time_data)

            else:
                # open new position
                self.open_position(time_stamp, time_data)






    def analyze_trade(self):
        # 计算总收益
        total_profit = self.current_capital - self.initial_capital
        total_return = (self.current_capital / self.initial_capital - 1)

        print(f"初始资金: {self.initial_capital}")
        print(f"最终资金: {self.current_capital}")
        print(f"总收益: {total_profit}")
        print(f"总收益率: {total_return * 100:.2f}%")
        # 获取时长
        print(f"APR: {100*(1 + total_return * (24 * 365 / len(self.time_stamps))):.2f}%")
        print(f"APY: {100*((1 + total_return) ** (24 * 365 / len(self.time_stamps)) - 1):.2f}%")
        trade_profits = pd.DataFrame(self.trade_profits)
        trade_positions = pd.DataFrame(self.trade_positions)
        self.trade_trails = pd.merge(trade_profits, trade_positions, on='current_time')
        self.trade_trails['return_per_time'] = self.trade_trails['pnl']/self.initial_capital
        self.trade_trails['curve'] = (self.trade_trails['pnl'].expanding().sum() + self.initial_capital)/self.initial_capital


        self.trade_trails['max2here'] = self.trade_trails['curve'].expanding().max()
        # 计算到历史最高值到当日的跌幅,draw-down
        self.trade_trails['dd2here'] = self.trade_trails['curve'] / self.trade_trails['max2here'] - 1
        # 计算最大回撤,以及最大回撤结束时间
        end_date, max_draw_down = tuple(self.trade_trails.sort_values(by=['dd2here']).iloc[0][['current_time', 'dd2here']])
        # 计算最大回撤开始时间
        start_date = self.trade_trails[self.trade_trails['current_time'] <= end_date].sort_values(by='curve', ascending=False).iloc[0][
            'current_time']
        sharpe_ratio = self.trade_trails['return_per_time'].mean() / self.trade_trails['return_per_time'].std()
        # 计算其他统计数据
        print(f"sharpe_ratio: {sharpe_ratio*sqrt(365):.2f}")
        print(f'max_draw_down: {max_draw_down:.2f}')
        print(f'最大回撤开始时间:{start_date}')
        print(f'最大回撤结束时间:{end_date}')
        avg_profit_per_trade = trade_profits['pnl'].sum() / self.num_trades if self.num_trades > 0 else 0
        max_profit = max(trade_profits['pnl'])
        min_profit = min(trade_profits['pnl'])

        print(f"交易次数: {self.num_trades}")
        print(f"平均每笔交易收益: {avg_profit_per_trade:.2f}")
        print(f"最大单笔收益: {max_profit:.2f}")
        print(f"最小单笔收益: {min_profit:.2f}")

        self.trade_trails = pd.merge(self.trade_trails, self.target_data, left_on='current_time', right_on='snapshot_time', how='left')


        # 创建收益曲线图
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            specs=[
                [{"type": "xy", "secondary_y": True}],
                [{"type": "xy", "secondary_y": False}],
            ],
        )
        # 添加累积净值折线图
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['curve'],
                                 mode='lines+markers', name='累积净值'), row=1, col=1)
        # 添加累积净值折线图
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['btc_price']/self.trade_trails['btc_price'].loc[0],
                                 mode='lines+markers', name='btc_price'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['dd2here'],
                       mode='lines',
                       name='max_drawdown',
                       fill='tozeroy',  # fill参数设置为'tozeroy'表示填充到 y=0 的水平线
                       fillcolor='rgba(65,105,225,0.2)',  # 设置填充颜色和透明度
                       line={'color': 'rgba(65,105,225,0.2)', 'width': 1}),
            secondary_y=True, row=1, col=1,
        )
        # 调整每笔收益散点图以使其更加显眼，并在每个点上标明此单收益
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['pnl'],
                                 mode='markers+text', name='每笔收益', marker_color='green', opacity=1,),
                                 # text=[f"{round(value, 2)}" for value in self.trade_trails['pnl']],
                                 # textposition='top center',
                                 # textfont=dict(size=14, color='black')),
                                 row=2, col=1)

        # 更新布局
        fig.update_layout(
            title='收益曲线图',
            xaxis_title='时间',
            yaxis_title='累积净值',
            yaxis2_title='最大回撤',
            legend=dict(x=0, y=1.2, orientation='h')
        )
        import plotly.io as pio

        pio.renderers.default = 'browser'  # 或尝试其他渲染模式
        fig.show()



if __name__ == '__main__':
    data = pd.read_pickle('data/btc_option_data_for_trade.pkl')
    backtrader = BackTrader(initial_capital=60000,data=data, date_interval=['2024-01-23 00:00:00', '2024-07-27 00:00:00'], fraction=0.01, exe_price_gear=1, mature_gear=0)
    backtrader.trade()
    backtrader.analyze_trade()
