# -*- coding:utf-8 -*-
"""
@FileName：strategies.py
@Description：
@Author：fdguuwzj
@Time：2024/10/21 16:15
"""
import os
from math import sqrt

from config import BACKTEST_DIR, OUTPUT_DIR

# -*- coding:utf-8 -*-
"""
@FileName：get_trade_records.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

time_spread = pd.read_csv(os.path.join(BACKTEST_DIR, 'min_max_options_couple.csv'))
time_spread['short_box_mature'] = pd.to_datetime(time_spread['short_box_mature'])
time_spread['long_box_mature'] = pd.to_datetime(time_spread['long_box_mature'])
time_spread['snapshot_time'] = pd.to_datetime(time_spread['snapshot_time'])


class Option:
    def __init__(self, snapshot_time: str = '0',
                 name: str = '0', ask_price: float = 0,
                 bid_price: float = 0, mark_price: float = 0,
                 expiry_date: pd.Timestamp = 0, exe_price: float = 0.0,
                 type: str = 'C', btc_price: float = 0.0, side: str = 'buy', number: float = 1):
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
        self.number : float = number # 持仓数量，最小为0.1张


class Position:
    def __init__(self, current_time: pd.Timestamp, options: list[Option] = None):
        self.current_time = current_time
        self.options: list[Option] = options if options else None
        try:
            self.expiry_date = min([option.expiry_date for option in options]) if options else None
            self.volume = sum([abs(option.number*option.mark_price*option.btc_price)  for option in options])  if options else 0
        except Exception as e:
            print(e)
    


class BackTrader:
    def __init__(self, initial_capital=30000, strategy_params={'name': 'sell_straddle', 'exe_price_gear': 1, 'mature_gear': 0},
                 data=None, fraction=0.001, portfolio_num=1, date_interval = ['2024-01-01 00:00:00', '2024-07-27 00:00:00']):
        # 创建一个字典来存储当前持仓

        self.current_position: Position=None
        # 记录所有的持仓信息
        self.trade_positions = []
        # 创建一个列表来存储每次交易的收益
        self.trade_profits = []
        # 交易次数
        self.num_trades = 0
        # 初始资金
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy_params = strategy_params
        # 输入数据
        self.data = data
        self.target_data =  self.process_target_data(data)
        self.time_stamps = data[(data['snapshot_time'] >= date_interval[0]) & (data['snapshot_time'] <= date_interval[1])]['snapshot_time'].unique()
        self.real_start_time = max(pd.to_datetime(date_interval[0]), self.data["snapshot_time"].min())
        self.real_end_time = min(pd.to_datetime(date_interval[1]), self.data["snapshot_time"].max())
        # 输出data数据的起讫日期
        logger.info(f'data数据的起始日期: {self.data["snapshot_time"].min()}, 结束日期: {self.data["snapshot_time"].max()}')
        logger.info(f'回测起始日期: {self.real_start_time}, 回测结束日期: {self.real_end_time}, 总共时长: {(self.real_end_time - self.real_start_time).total_seconds() / 3600} h')
        # 交易设置
        self.fraction = fraction
        self.portfolio_num = portfolio_num
        self.save_dir = os.path.join(OUTPUT_DIR, f'{strategy_params["name"]}_{self.real_start_time.strftime("%Y%m%d_%H%M%S")}_{self.real_end_time.strftime("%Y%m%d_%H%M%S")}')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


    def process_target_data(self, data):
        data1 = data[['snapshot_time', 'btc_price']].drop_duplicates().set_index('snapshot_time')
        data2 = data1.copy()
        data2.index = data1.index - pd.Timedelta(seconds=1)
        all_data = pd.concat([data1, data2])
        all_data.sort_values(by=['snapshot_time'], inplace=True)
        return all_data
    def sell(self, item, amount=1, opponent=False):

        # 进行盘口挂单
        if opponent:
            return item.bid_price*(1-self.fraction)*amount
        else:
            return item.mark_price * (1 + self.fraction)*amount

    def buy(self, item, amount=1, opponent=False):

        # 进行盘口挂单
        if opponent:
            return item.ask_price*(1 + self.fraction)*amount
        else:
            return item.mark_price * (1 + self.fraction)*amount


    def arraive_time(self, current_time: pd.Timestamp):
        # 如果当前时间大于持仓的到期时间，平仓
        if  not hasattr(self.current_position, 'expiry_date'):
            return True
        elif current_time >= self.current_position.expiry_date:
        # elif current_time >= self.current_position.expiry_date - pd.Timedelta(hours=8):
            return True
        else:
            return False

    def get_position(self, position : Position, alter_time=None):
        position_dict = vars(position)
        infos = position_dict.copy()
        infos['options'] = [_.name for _ in position_dict['options']]
        if alter_time:
            infos['current_time'] = alter_time
        return infos
    def close_position(self, current_time: pd.Timestamp):
        # 持仓 -》 空仓

        now_btc_price = self.target_data.loc[current_time, 'btc_price']
        # 只考虑到期行权情况
        position_pnl = 0
        for option in self.current_position.options:
            if option.side == 'buy' and option.type == 'C':
                position_pnl += max(now_btc_price - option.exe_price, 0)*option.number
            elif option.side == 'sell' and option.type == 'C':
                position_pnl += min(-now_btc_price + option.exe_price, 0)*option.number
            elif option.side == 'buy' and option.type == 'P':
                position_pnl += max(-now_btc_price + option.exe_price , 0)*option.number
            elif option.side == 'sell' and option.type == 'P':
                position_pnl += min(now_btc_price - option.exe_price, 0)*option.number
        # 为方便记账，这里将时间向前调整1s
        self.trade_profits.append({'current_time': current_time-pd.Timedelta(seconds=1), 'pnl': position_pnl, 'pnl_type': 'return'})

        self.trade_positions.append(self.get_position(self.current_position, alter_time=current_time-pd.Timedelta(seconds=1)))
        self.current_capital += position_pnl
        self.current_position = Position(current_time)
        logger.info(f'{current_time}: close_position, pnl = {position_pnl}, current_capital: {self.current_capital}')


    def after_open_position(self, current_time, position_pnl, btc_price):
            logger.info(f'position opened:{current_time}, btc_price = {btc_price}')
            for option in self.current_position.options:
                logger.info(f'open {option.side} {option.number} {option.name} at {option.mark_price}btc')
            self.trade_profits.append({'current_time': current_time, 'pnl': position_pnl, 'pnl_type': 'premium'})
            self.current_capital += position_pnl
            self.trade_positions.append(self.get_position(self.current_position))
            self.num_trades += 1
            logger.info(f'option premium: {position_pnl} current_capital: {self.current_capital}')

    def open_sell_straddle_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓 
        # 开近月平值双卖
        mature_date = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear'])
        time_options = time_options[time_options['expiry_date'] == mature_date]
        btc_price = time_options['btc_price'].values[0]
        exe_prices = sorted(time_options['exe_price'].astype('float').unique())
        down_exe_price, up_exe_price = self.get_exe_price(btc_price, exe_prices, self.strategy_params['exe_price_gear'])
        down_put = self.extract_option(time_options, down_exe_price, 'P', 'sell', self.portfolio_num)
        up_call = self.extract_option(time_options, up_exe_price, 'C', 'sell', self.portfolio_num)
        if down_put and up_call:
            self.current_position = Position(current_time, [down_put, up_call])
            position_pnl =  self.sell(down_put, self.portfolio_num)*down_put.btc_price + self.sell(up_call, self.portfolio_num)*up_call.btc_price
            self.after_open_position(current_time, position_pnl, btc_price)
        else:
            logger.warning(f'{current_time} miss data, down_put is {down_put}, up_call is {up_call}, skipped')

    def open_buy_straddle_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        # 开近月平值双卖

        mature_date = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear'])
        time_options = time_options[time_options['expiry_date'] == mature_date]
        btc_price = time_options['btc_price'].values[0]
        exe_prices = sorted(time_options['exe_price'].astype('float').unique())
        down_exe_price, up_exe_price = self.get_exe_price(btc_price, exe_prices, self.strategy_params['exe_price_gear'])
        down_put = self.extract_option(time_options, down_exe_price, 'P', 'buy', self.portfolio_num)
        up_call = self.extract_option(time_options, up_exe_price, 'C', 'buy', self.portfolio_num)
        if down_put and up_call:
            self.current_position = Position(current_time, [down_put, up_call])
            position_pnl = - self.buy(down_put, self.portfolio_num)*down_put.btc_price - self.buy(up_call, self.portfolio_num)*up_call.btc_price
            self.after_open_position(current_time, position_pnl, btc_price)
        else:
            logger.warning(f'{current_time} miss data, down_put is {down_put}, up_call is {up_call}, skipped')



    def open_buy_calender_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        # 开近月平值双卖 开远月平值双买
        btc_price = time_options['btc_price'].values[0]

        mature_date1 = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear1'])
        time_options1 = time_options[time_options['expiry_date'] == mature_date1]
        exe_prices1 = sorted(time_options1['exe_price'].astype('float').unique())
        down_exe_price1, up_exe_price1 = self.get_exe_price(btc_price, exe_prices1, self.strategy_params['exe_price_gear1'])
        down_put1 = self.extract_option(time_options1, down_exe_price1, 'P', 'sell', self.portfolio_num)
        up_call1 = self.extract_option(time_options1, up_exe_price1, 'C', 'sell', self.portfolio_num)

        mature_date2 = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear2'])
        time_options2 = time_options[time_options['expiry_date'] == mature_date2]
        exe_prices2 = sorted(time_options2['exe_price'].astype('float').unique())
        down_exe_price2, up_exe_price2 = self.get_exe_price(btc_price, exe_prices2, self.strategy_params['exe_price_gear2'])
        down_put2 = self.extract_option(time_options2, down_exe_price2, 'P', 'buy', self.portfolio_num)
        up_call2 = self.extract_option(time_options2, up_exe_price2, 'C', 'buy', self.portfolio_num)
        if down_put1 and up_call1 and down_put1 and up_call1:
            self.current_position = Position(current_time, [down_put1, up_call1, down_put2, up_call2])
            position_pnl =  self.sell(down_put1, self.portfolio_num)*down_put1.btc_price + self.sell(up_call1, self.portfolio_num)*up_call1.btc_price  - self.buy(down_put2, self.portfolio_num)*down_put2.btc_price - self.buy(up_call2, self.portfolio_num)*up_call2.btc_price
            self.after_open_position(current_time, position_pnl, btc_price)
        else:
            logger.warning(f'{current_time} miss data, options are {[down_put1, up_call1, down_put2, up_call2]}')


    def open_time_straddle_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        # 开近月平值双卖 开远月平值双买
        btc_price = time_options['btc_price'].values[0]

        mature_date1 = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear1'])
        time_options1 = time_options[time_options['expiry_date'] == mature_date1]
        exe_prices1 = sorted(time_options1['exe_price'].astype('float').unique())
        down_exe_price1, up_exe_price1 = self.get_exe_price(btc_price, exe_prices1, self.strategy_params['exe_price_gear1'])
        down_put1 = self.extract_option(time_options1, down_exe_price1, 'P', 'sell', self.portfolio_num)
        up_call1 = self.extract_option(time_options1, up_exe_price1, 'C', 'sell', self.portfolio_num)

        mature_date2 = self.get_mature_date(time_options, current_time, self.strategy_params['mature_gear2'])
        time_options2 = time_options[time_options['expiry_date'] == mature_date2]
        exe_prices2 = sorted(time_options2['exe_price'].astype('float').unique())
        down_exe_price2, up_exe_price2 = self.get_exe_price(btc_price, exe_prices2, self.strategy_params['exe_price_gear2'])
        down_put2 = self.extract_option(time_options2, down_exe_price2, 'P', 'buy', self.portfolio_num)
        up_call2 = self.extract_option(time_options2, up_exe_price2, 'C', 'buy', self.portfolio_num)
        if down_put1 and up_call1 and down_put1 and up_call1:
            self.current_position = Position(current_time, [down_put1, up_call1, down_put2, up_call2])
            position_pnl =  self.sell(down_put1, self.portfolio_num)*down_put1.btc_price + self.sell(up_call1, self.portfolio_num)*up_call1.btc_price  - self.buy(down_put2, self.portfolio_num)*down_put2.btc_price - self.buy(up_call2, self.portfolio_num)*up_call2.btc_price
            self.after_open_position(current_time, position_pnl, btc_price)
        else:
            logger.warning(f'{current_time} miss data, options are {[down_put1, up_call1, down_put2, up_call2]}')

    def open_buy_call_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        pass


    def open_sell_butterfly_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓 
        pass

    def open_buy_butterfly_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        pass



    def open_sell_put_position(self, current_time: pd.Timestamp, time_options: pd.DataFrame):
        # 空仓 -》 持仓
        pass

    def extract_option(self, time_options: pd.DataFrame, exe_price: float, option_type: str = 'C', side: str = 'buy', num: float = 1):
        """
        选出目标期权
        :param time_options: 当前时间的所有期权数据
        :param exe_price: 行权价
        :param option_type: 期权类型
        :param side: 方向
        :param num: 数量
        :return:
        """
        option_row = time_options[(time_options['exe_price'].astype('float') == exe_price)& (time_options['type'] == option_type)]
        try:
            config = {
                'snapshot_time': option_row['snapshot_time'].values[0],
                'name': option_row['instrument_name'].values[0],
                'mark_price': option_row['mark_price'].values[0],
                'bid_price': option_row['bid_price'].values[0],
                'ask_price': option_row['ask_price'].values[0],
                'expiry_date': option_row['expiry_date'].values[0],
                'exe_price': float(option_row['exe_price'].values[0]),
                'type': option_row['type'].values[0],
                'btc_price': option_row['btc_price'].values[0],
            }
            return Option(snapshot_time=config['snapshot_time'], name=config['name'], mark_price=config['mark_price'],
                          bid_price=config['bid_price'], ask_price=config['ask_price'], exe_price=config['exe_price'],
                          expiry_date=config['expiry_date'], type=config['type'], btc_price=config['btc_price'],
                          side=side, number = num)
        except Exception as e:
            return None
            print(e)


    def get_mature_date(self, time_options, current_time, mature_gear):
        mature_dates = sorted([_ for _ in time_options['expiry_date'].unique() if _ > current_time])
        # 开近月期权
        return mature_dates[mature_gear]

    def get_exe_price(self, btc_price: float, exe_prices: list, exe_price_gear: int = 0):
        # 设置参数n，找出第n大的数，和第n小的数
        lower_exe_prices = sorted([exe_price for exe_price in exe_prices if exe_price < btc_price], reverse=True)[:exe_price_gear]
        upper_exe_prices = sorted([exe_price for exe_price in exe_prices if exe_price > btc_price], reverse=False)[:exe_price_gear]
        min_exe_price = lower_exe_prices[-1] if lower_exe_prices else None
        max_exe_price = upper_exe_prices[-1] if upper_exe_prices else None
        return min_exe_price, max_exe_price
    

    def open_position(self, time_stamp, time_data):
        if self.strategy_params['name'] == 'sell_straddle':
            self.open_sell_straddle_position( time_stamp, time_data)
        elif self.strategy_params['name'] == 'buy_straddle':
            self.open_buy_straddle_position( time_stamp, time_data)
        elif self.strategy_params['name'] == 'sell_butterfly':
            self.open_sell_butterfly_position( time_stamp, time_data)
        elif self.strategy_params['name'] == 'buy_butterfly':
            self.open_buy_butterfly_position( time_stamp, time_data)
        elif self.strategy_params['name'] == 'sell_put':
            self.open_sell_put_position(time_stamp, time_data)
        elif self.strategy_params['name'] == 'buy_call':
            self.open_buy_call_position(time_stamp, time_data)
        elif self.strategy_params['name'] == 'time_straddle':
            self.open_time_straddle_position(time_stamp, time_data)
        else:
            logger.error(f'strategy did not realize.')
    def trade(self):
        for time_stamp in self.time_stamps:
            current_time = pd.to_datetime(time_stamp)
            time_data = self.data[self.data['snapshot_time'] == current_time]
            if not self.empty_position():
                if self.arraive_time(time_stamp):
                    # close position
                    self.close_position(time_stamp)
                    # open new position
                    self.open_position(time_stamp, time_data)

            else:
                # open new position
                self.open_position(time_stamp, time_data)


    def empty_position(self):
        if (self.current_position == None) or  (self.current_position.options == None):
            return True
        else:
            return False


    def analyze_trade(self):
        # 计算总收益
        total_profit = self.current_capital - self.initial_capital
        total_return = (self.current_capital / self.initial_capital - 1)

        print(f"初始资金: {self.initial_capital}")
        print(f"最终资金: {self.current_capital}")
        print(f"总收益: {total_profit}")
        print(f"总收益率: {total_return * 100:.2f}%")
        # 获取时长
        print(f"APR: {100*(1 + total_return * (24 * 365 / len(self.time_stamps)) - 1):.2f}%")
        print(f"APY: {100*((1 + total_return) ** (24 * 365 / len(self.time_stamps)) - 1):.2f}%")
        trade_profits = pd.DataFrame(self.trade_profits)
        trade_positions = pd.DataFrame(self.trade_positions)

        self.trade_trails = pd.merge(trade_profits, trade_positions, on='current_time',how='outer')
        self.trade_trails.sort_values(by='current_time', inplace=True)
        self.trade_trails['return_per_time'] = self.trade_trails['pnl']/self.initial_capital
        self.trade_trails['curve'] = (self.trade_trails['pnl'].expanding().sum() + self.initial_capital)/self.initial_capital
        self.trade_trails['current_capital'] = self.trade_trails['pnl'].expanding().sum() + self.initial_capital


        self.trade_trails['max2here'] = self.trade_trails['curve'].expanding().max()
        # 计算到历史最高值到当日的跌幅,draw-down
        self.trade_trails['dd2here'] = self.trade_trails['curve'] / self.trade_trails['max2here'] - 1
        # 计算最大回撤,以及最大回撤结束时间
        end_date, max_draw_down = tuple(self.trade_trails.sort_values(by=['dd2here']).iloc[0][['current_time', 'dd2here']])
        # 计算最大回撤开始时间
        start_date = self.trade_trails[self.trade_trails['current_time'] <= end_date].sort_values(by='curve', ascending=False).iloc[0][
            'current_time']
        sharpe_ratio = self.trade_trails['return_per_time'].mean() / self.trade_trails['return_per_time'].std()
        daily_turnover = (self.trade_trails['volume']/ self.trade_trails['current_capital']).sum() / (self.real_end_time - self.real_start_time).days

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
        print(f"daily turnover: {daily_turnover:.2f}")
        self.target_data['btc_volatility'] = self.target_data['btc_price'].pct_change(1).rolling(15*24).std()
        self.trade_trails = pd.merge(self.trade_trails, self.target_data, left_on='current_time', right_on='snapshot_time', how='left')
        self.trade_trails.to_csv(f'{self.save_dir}/trade_trails.csv', index=False)

        # 创建收益曲线图
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            specs=[
                [{"type": "xy", "secondary_y": True}],
                [{"type": "xy", "secondary_y": False}],
                [{"type": "xy", "secondary_y": False}],
                [{"type": "table"}],
            ],
        )
        # 添加累积净值折线图
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['curve'],
                                 mode='lines+markers', name='累积净值'), row=1, col=1)
        # 添加累积净值折线图
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['btc_price']/self.trade_trails['btc_price'].loc[0],
                                 mode='lines+markers', name='btc_price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.trade_trails['current_time'], y=self.trade_trails['btc_volatility'],
                                 mode='lines+markers', name='btc_volatility'), row=3, col=1)
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
                                 mode='markers+text', name='每笔收益', marker_color=self.trade_trails['pnl_type'].map({'premium': 'green', 'return': 'red'}), opacity=1,),
                                 row=2, col=1)
        # 添加一条横线，y值为-1500
        fig.add_trace(go.Scatter(x=[self.trade_trails['current_time'].min(), self.trade_trails['current_time'].max()], y=[-1500, -1500],
                                 mode='lines', name='-1500', line=dict(color='blue', width=2)), row=2, col=1)

        # 添加统计信息到表格
        stats_table = go.Table(
            header=dict(values=["初始资金", "最终资金", "总收益", "总收益率(%)", "APR(%)", "APY(%)", "annual sharpe", "最大回撤(%)", "最大回撤开始时间", "最大回撤结束时间"]),
            cells=dict(values=[
    
                round(self.initial_capital, 2), round(self.current_capital, 2), round(total_profit, 2), round(total_return * 100, 2),
                round(100 * (1 + total_return * (24 * 365 / len(self.time_stamps)) - 1), 2), 
                round(100 * ((1 + total_return) ** (24 * 365 / len(self.time_stamps)) - 1), 2), 
                round(sharpe_ratio * sqrt(365), 2), round(max_draw_down*100, 2), start_date, end_date
            ]),
            columnwidth=[100, 100] * 11  # 设置列宽
        )
        fig.add_trace(stats_table, row=4, col=1)

        # 更新布局
        fig.update_layout(
            title=f'{self.strategy_params["name"]}收益曲线图 {self.strategy_params.items()}',
            xaxis_title='时间',
            yaxis_title='累积净值',
            yaxis2_title='最大回撤',
            legend=dict(x=0, y=1.2, orientation='h')
        )
        import plotly.io as pio

        pio.renderers.default = 'browser'  # 或尝试其他渲染模式

        pio.write_html(fig, f'{self.strategy_params["name"]}收益曲线图 {self.strategy_params.items()}.html')

        fig.show()

    def analyze_btc(self):

        self.target_data['btc_volatility'] = self.target_data['btc_price'].pct_change(1).rolling(15*24).std()


        # 创建收益曲线图
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            specs=[
                [{"type": "xy", "secondary_y": True}],
            ],
        )

        fig.add_trace(go.Scatter(x=self.target_data['current_time'], y=self.target_data['btc_price']/self.target_data['btc_price'].loc[0],
                                 mode='lines+markers', name='btc_price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.target_data['current_time'], y=self.target_data['btc_volatility'],
                                 mode='lines+markers', name='btc_volatility'),secondary_y=True, row=1, col=1)

        # 更新布局
        fig.update_layout(
            title=f'{self.strategy_params["name"]}收益曲线图 {self.strategy_params.items()}',
            xaxis_title='时间',
            yaxis_title='累积净值',
            legend=dict(x=0, y=1.2, orientation='h')
        )
        import plotly.io as pio

        pio.renderers.default = 'browser'  # 或尝试其他渲染模式
        fig.show()


class BoxSpreadBackTrader:
    def __init__(self, current_position={}, trade_positions=[], trade_profits=[], initial_capital=30000,
                 time_spread=time_spread):
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

    def rebalance_box_spread_position(self, row):
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

    def rebalance_dual_sell_position(self, row):
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
            self.rebalance_box_spread_position(row)
            # logger.info(f"开仓: {self.current_position}")
        else:
            if self.arraive_time(row['snapshot_time']):
                # 平仓
                self.rebalance_box_spread_position(row)
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
        print(f"年化收益率: {total_return ** (24 * 365 / len(self.time_spread)):.2f}%")
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
                                 text=[f"{round(value, 2)}" for value in self.trade_trails['current_spread']],
                                 textposition='top center',
                                 textfont=dict(size=14, color='black')), row=2, col=1)

        # 更新布局
        fig.update_layout(
            title=f'收益曲线图',
            xaxis_title='时间',
            yaxis_title='累积净值',
            yaxis2_title='单笔收益',
            legend=dict(x=0, y=1.2, orientation='h')
        )

        fig.show()


