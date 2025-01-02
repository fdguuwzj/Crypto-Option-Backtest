# -*- coding:utf-8 -*-
"""
@FileName：plotOptions.py
@Description：
@Author：fdguuwzj
@Time：2024-03-14 22:22
"""
from options_utils import *
import plotly.graph_objects as go
# from pyecharts import options as opts
# from pyecharts.charts import Line
# #
# # # 定义期权盈亏函数
# # def option_profit_loss(S, K, premium, quantity):
# #     """
# #     S: 股票价格
# #     K: 期权行权价
# #     premium: 期权费用
# #     quantity: 期权数量
# #     """
# #     # 计算期权盈亏
# #     profit_loss = (S - K) * quantity - premium * quantity
# #     return profit_loss
# #
# # # 定义股票价格范围
# # stock_price_range = range(80, 120)
# #
# # # 定义期权参数
# # strike_price = 100
# # option_premium = 3
# # option_quantity = 1
# #
# # # 计算盈亏数据
# # profit_loss_data = [option_profit_loss(S, strike_price, option_premium, option_quantity) for S in stock_price_range]
# #
# # # 创建线性图
# # line_chart = Line()
# # line_chart.add_xaxis(list(stock_price_range))
# # line_chart.add_yaxis("盈亏", profit_loss_data, is_smooth=True)
# #
# # # 设置图表标题和坐标轴标签
# # line_chart.set_global_opts(
# #     title_opts=opts.TitleOpts(title="期权盈亏分析图"),
# #     xaxis_opts=opts.AxisOpts(name="股票价格"),
# #     yaxis_opts=opts.AxisOpts(name="盈亏")
# # )
# #
# # # 渲染并保存图表
# # line_chart.render("option_profit_loss.html")
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# def option_profit_loss(S, K, premium, quantity, option_type):
#     """
#     S: 标的资产价格
#     K: 期权行权价
#     premium: 期权费用
#     quantity: 期权数量
#     option_type: 期权类型，'call'表示看涨期权，'put'表示看跌期权
#     """
#     if option_type == 'call':
#         payoff = np.maximum(S - K, 0) - premium
#     elif option_type == 'put':
#         payoff = np.maximum(K - S, 0) - premium
#     else:
#         raise ValueError("Invalid option type")
#
#     profit_loss = payoff * quantity
#     return profit_loss
#
# # 定义标的资产价格范围
# S_range = np.linspace(20, 120, 100)
#
# # 定义期权参数
# strike_price = 100
# option_premium = 3
# option_quantity = 1
# option_type = 'call'
#
# # 计算盈亏数据
# profit_loss_data = option_profit_loss(S_range, strike_price, option_premium, option_quantity, option_type)
#
# # 创建线性图
# line_chart = Line()
# line_chart.add_xaxis(S_range)
# line_chart.add_yaxis("盈亏", profit_loss_data, is_smooth=True)
#
# # 设置图表标题和坐标轴标签
# line_chart.set_global_opts(
#     title_opts=opts.TitleOpts(title="期权盈亏分析图"),
#     xaxis_opts=opts.AxisOpts(name="股票价格"),
#     yaxis_opts=opts.AxisOpts(name="盈亏")
# )
# # 渲染并保存图表
# line_chart.render("option_profit_loss.html")

import numpy as np
import matplotlib.pyplot as plt

# ===备兑开仓
# 买stock 卖call

# X = 20
# c = 5
time2mature = 31 / 365
r = 0.1
iv = 0.5
S = np.linspace(45000, 95000, 100)


def show_protfolio(title, S, option1_exc, option1_price, option1_type, option1_trade, option2_exc, option2_price,
                   option2_type, option2_trade, r, iv, time2mature, future_price=0, option1_weight=1, option2_weight=1,
                   future_weight=0):
    """

    :param title: 图片标题
    :param S: 标的价格范围
    :param option1_exc: 期权1的行权价
    :param option1_price: 期权1的价格
    :param option1_type: 期权1的类型 'call' or 'put'
    :param option1_trade: 期权1的交易类型 'buy' or 'sell'
    :param option2_exc: 期权2的行权价
    :param option2_price: 期权2的价格
    :param option2_type:期权2的类型 'call' or 'put'
    :param option2_trade: 期权2的交易类型  'buy' or 'sell'
    :param r: 无风险利率
    :param iv: 隐波
    :param time2mature: 剩余到期时间（年）
    :param future_price: 期货价格
    :param option1_weight: 期权1的权重
    :param option2_weight: 期权2的权重
    :param future_weight: 期货的权重
    :return:
    """
    # 创建折线图轨迹
    option1 = vanilla_option_return(S, option1_exc, time2mature, r, iv, option1_price, 0, q=0, trade_type=option1_trade,
                                    option_type=option1_type)
    option2 = vanilla_option_return(S, option2_exc, time2mature, r, iv, option2_price, 0, q=0, trade_type=option2_trade,
                                    option_type=option2_type)
    option1_m = vanilla_option_return(S, option1_exc, 0, r, iv, option1_price, 0, q=0, trade_type=option1_trade,
                                      option_type=option1_type)
    option2_m = vanilla_option_return(S, option2_exc, 0, r, iv, option2_price, 0, q=0, trade_type=option2_trade,
                                      option_type=option2_type)
    trace_option1 = go.Scatter(x=S, y=option1_m, mode='lines', name='期权1到期收益')
    trace_option2 = go.Scatter(x=S, y=option2_m, mode='lines', name='期权2到期收益')
    trace1 = go.Scatter(x=S,
                        y=(option1 * option1_weight + option2 * option2_weight + future_weight * (S - future_price)),
                        mode='lines', name='期权组合当期预估收益')
    trace2 = go.Scatter(x=S, y=(
                option1_m * option1_weight + option2_m * option2_weight + future_weight * (S - future_price)),
                        mode='lines', name='期权组合到期收益')
    data_list = [trace_option1, trace_option2, trace1, trace2]
    if future_weight != 0:
        trace_future = go.Scatter(x=S, y=future_weight * (S - future_price), mode='lines', name='标的物到期收益')
        data_list.append(trace_future)
    # 创建图表布局
    layout = go.Layout(title=title, xaxis=dict(title='标的价格'), yaxis=dict(title='收益'))
    # 创建图表对象
    fig = go.Figure(data=data_list, layout=layout)
    # 显示图表
    fig.show()


def show_protfolio_list(title, S, options, r, iv, future_price=0, future_weight=0):
    """
    :param title: 图片标题
    :param S: 标的价格范围
    :param options: list[dict]
    :param r: 无风险利率
    :param iv: 隐波
    :param future_price: 期货价格
    :param future_weight: 期货的权重
    :return:
    """
    # 创建折线图轨迹
    option_trace_now = 0
    option_trace_muture = 0
    data_list = []
    for id, option in enumerate(options):
        option1_exc, option1_price, option1_trade, option1_type, time2mature, option1_weight = option['exc'], option[
            'price'], option['trade'], option['type'], option['time2mature'], option['weight']
        option1 = vanilla_option_return(S, option1_exc, time2mature, r, iv, option1_price, 0, q=0,
                                        trade_type=option1_trade, option_type=option1_type)
        option1_m = vanilla_option_return(S, option1_exc, 0, r, iv, option1_price, 0, q=0, trade_type=option1_trade,
                                          option_type=option1_type)
        trace_option1 = go.Scatter(x=S, y=option1_m, mode='lines', name='期权1到期收益')
        data_list.append(trace_option1)
        option_trace_now += option1 * option1_weight
        option_trace_muture += option1_m * option1_weight
    trace1 = go.Scatter(x=S, y=(option_trace_now + future_weight * (S - future_price)), mode='lines',
                        name='期权组合当期预估收益')
    trace2 = go.Scatter(x=S, y=(option_trace_muture + future_weight * (S - future_price)), mode='lines',
                        name='期权组合到期收益')
    data_list += [trace1, trace2]
    if future_weight != 0:
        trace_future = go.Scatter(x=S, y=future_weight * (S - future_price), mode='lines', name='标的物到期收益')
        data_list.append(trace_future)
    # 创建图表布局
    layout = go.Layout(title=title, xaxis=dict(title='标的价格'), yaxis=dict(title='收益'))
    # 创建图表对象
    fig = go.Figure(data=data_list, layout=layout)
    # 显示图表
    fig.show()




# show_protfolio('双买', S, 64000, 0.0625*target_price, 'put', 'buy', 64000, 0.0539*target_price, 'call', 'buy', r, iv, time2mature)
# show_protfolio('双卖', S, 64000, 0.0625*target_price, 'put', 'sell', 64000, 0.0539*target_price, 'call', 'sell', r, iv, time2mature)
# show_protfolio('牛市认购价差', S,
#                [[62000, 0.0699 * target_price, 'call', 'buy'], [64000, 0.0544 * target_price, 'call', 'sell']], r, iv,
#                time2mature)
# show_protfolio('牛市认沽价差', S,
#                [[62000, 0.0464 * target_price, 'put', 'buy'], [64000, 0.0625 * target_price, 'put', 'sell']], r, iv,
#                time2mature)
# show_protfolio('熊市认购价差', S,
#                [[62000, 0.0699 * target_price, 'call', 'sell'], [64000, 0.0544 * target_price, 'call', 'buy']], r, iv,
#                time2mature)
# show_protfolio('熊市认沽价差', S,
#                [[62000, 0.0452 * target_price, 'put', 'buy'], [58000, 0.0226 * target_price, 'put', 'sell']], r, iv,
#                time2mature)
# show_protfolio('领口价差', S, 66000, 2800, 'put', 'buy', 73000, 1800, 'call', 'sell', r, iv, time2mature, future_price=67000, option1_weight=1, option2_weight=1, future_weight=0)
# show_protfolio('领口价差', S, 66000, 2800, 'put', 'buy', 73000, 1800, 'call', 'sell', r, iv, time2mature, future_price=67000, option1_weight=1, option2_weight=1, future_weight=1)

# show_protfolio('call_put_parity', S, 64000, 0.0625*target_price, 'put', 'buy', 64000, 0.0539*target_price, 'call', 'sell', r, iv, time2mature, future_price=target_price, option1_weight=1, option2_weight=1, future_weight=1)

if __name__ == '__main__':
    btc_price = 64000
    options = [
        {
            'exc': 53000
            , 'price': 0.011*btc_price
            , 'trade': 'buy'
            , 'type': 'put'
            , 'time2mature': 30
            , 'weight': 14/36.4
        },
        {
            'exc': 52000
            , 'price': 0.0095*btc_price
            , 'trade': 'buy'
            , 'type': 'put'
            , 'time2mature': 30
            , 'weight': 2.4/36.4
        },
        {
            'exc': 65000
            , 'price': 0.088*btc_price
            , 'trade': 'sell'
            , 'type': 'put'
            , 'time2mature': 60
            , 'weight': 10/36.4
        },
        {
            'exc': 48000
            , 'price': 0.013*btc_price
            , 'trade': 'buy'
            , 'type': 'put'
            , 'time2mature': 60
            , 'weight': 10/36.4
        },

    ]
    show_protfolio_list('portfolio', S, options, r=r, iv=iv, future_price=btc_price)
