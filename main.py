# -*- coding:utf-8 -*-
"""
@FileName：main.py
@Description：
@Author：fdguuwzj
@Time：2024-09-26 11:35
"""
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from strategies import BackTrader
from utils.options_utils import btc_vol_df

if __name__ == '__main__':
    # backtrader = BoxSpreadBackTrader()
    # backtrader.trade()
    # backtrader.analyze_trade()

    # 双卖回测
    data = pd.read_pickle('data/btc_option_data_for_trade1105.pkl')
    # 统计每天的数据条数
    # daily_counts = data.groupby(data['snapshot_time'].dt.date).size()
    # print('每日数据条数统计:')
    # print(daily_counts)
    # backtrader = BackTrader(initial_capital=60000, strategy='buy_straddle' ,data=data, date_interval=['2024-01-23 00:00:00', '2024-11-05 00:00:00'], fraction=0.001, exe_price_gear=1, mature_gear=0)
    # backtrader.trade()
    # backtrader.analyze_trade()

    # backtrader_buy_straddle = BackTrader(initial_capital=60000, strategy='buy_straddle' ,data=data, date_interval=['2024-01-23 00:00:00', '2024-11-05 00:00:00'], fraction=0.001, exe_price_gear=1, mature_gear=0)
    # backtrader_buy_straddle.trade()
    # backtrader_buy_straddle.analyze_trade()
    # backtrader_sell_straddle = BackTrader(initial_capital=60000, strategy='sell_straddle' ,data=data, date_interval=['2024-01-23 00:00:00', '2024-11-05 00:00:00'], fraction=0.001, exe_price_gear=1, mature_gear=0)
    # backtrader_sell_straddle.trade()
    # backtrader_sell_straddle.analyze_trade()
    #
    # buy_trade_trails = backtrader_buy_straddle.trade_trails[['current_time','curve']]
    # sell_trade_trails = backtrader_sell_straddle.trade_trails[['current_time','curve']]
    # buy_trade_trails.to_csv(r'buy_trade_trails1105.csv')
    # sell_trade_trails.to_csv(r'sell_trade_trails1105.csv')
    buy_trade_trails = pd.read_csv(r'buy_trade_trails1105.csv')
    sell_trade_trails = pd.read_csv(r'sell_trade_trails1105.csv')


    threshold = 0.0075

    btc_price = btc_vol_df()[['candle_begin_time', 'open', 'btc_volatility']].rename(columns={'candle_begin_time': 'current_time', 'open': 'btc_price'})
    buy_trade_trails.drop_duplicates(subset='current_time', keep='first', inplace=True)
    sell_trade_trails.drop_duplicates(subset='current_time', keep='first', inplace=True)
    buy_trade_trails['curve_delta'] = buy_trade_trails['curve'].astype(np.float64).pct_change()
    buy_trade_trails['curve_delta'].fillna(0, inplace=True)
    sell_trade_trails['curve_delta'] = sell_trade_trails['curve'].astype(np.float64).pct_change()
    sell_trade_trails['curve_delta'].fillna(0, inplace=True)
    all_trade_trails = pd.merge(buy_trade_trails, sell_trade_trails, on = 'current_time', suffixes = ['_x', '_y'])
    all_trade_trails['current_time'] = pd.to_datetime(all_trade_trails['current_time'])
    all_trade_trails = pd.merge(all_trade_trails, btc_price, on = 'current_time')
    all_trade_trails['curve_delta'] = np.where(all_trade_trails['btc_volatility'] >= threshold, all_trade_trails['curve_delta_y'], all_trade_trails['curve_delta_x'])
    all_trade_trails['curve'] = (1+all_trade_trails['curve_delta']).cumprod()
    all_trade_trails['max2here'] = all_trade_trails['curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅,draw-down
    all_trade_trails['dd2here'] = all_trade_trails['curve'] / all_trade_trails['max2here'] - 1
    import plotly.graph_objects as go

    # 根据波动率条件分别绘制红色和蓝色曲线
    high_vol_mask = all_trade_trails['btc_volatility'] >= threshold
    
    # 分别获取高波动率和低波动率的数据点
    high_vol_data = all_trade_trails[high_vol_mask]
    low_vol_data = all_trade_trails[~high_vol_mask]
    
    # 创建图形
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        specs=[
            [{"type": "xy", "secondary_y": True}],

        ],
    )
    
    # 添加高波动率散点 - 使用主轴
    fig.add_trace(go.Scatter(
        x=high_vol_data['current_time'],
        y=high_vol_data['curve'],
        mode='markers',
        name='High Volatility Curve',
        marker=dict(color='red', size=5)
    ))
    
    # 添加低波动率散点 - 使用主轴
    fig.add_trace(go.Scatter(
        x=low_vol_data['current_time'],
        y=low_vol_data['curve'],
        mode='markers',
        name='Low Volatility Curve',
        marker=dict(color='blue', size=5),
    ),row=1, col=1,)

    # 添加最大回撤曲线 - 使用副轴
    fig.add_trace(
        go.Scatter(x=all_trade_trails['current_time'], y=all_trade_trails['dd2here'],
                   mode='lines',
                   name='max_drawdown',
                   fill='tozeroy',  # fill参数设置为'tozeroy'表示填充到 y=0 的水平线
                   fillcolor='rgba(65,105,225,0.2)',  # 设置填充颜色和透明度
                   line={'color': 'rgba(65,105,225,0.2)', 'width': 1}),
        secondary_y=True, row=1, col=1,
    )
    
    # 更新布局
    fig.update_layout(
        title=f'波动率阈值为{threshold}时，资金曲线与最大回撤 (红色: 高波动率, 蓝色: 低波动率)',
        xaxis_title='时间',
        yaxis_title='净值',
        yaxis2=dict(
            title='回撤',
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )
    import plotly.io as pio

    pio.renderers.default = 'browser'  # 或尝试其他渲染模式
    fig.show()
    
    print(all_trade_trails)



