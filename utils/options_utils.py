# -*- coding:utf-8 -*-
"""
@FileName：options_utils.py
@Description：fdguuwzj self-made option utils toolkit
@Author：fdguuwzj
@Time：2023-12-11 16:39
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
episode = 1e-16
# bs公式求期权价格
def vanilla_option(S, K, T, r, sigma, q=0, option_type='call'):
    """
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: standard deviation of price of underlying asset
    q: dividend rate
    option_type: 期权类型
    return 期权定价
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + episode)
    d2 = d1  - sigma * np.sqrt(T)
    if option_type == 'call' or option_type == 'C' or option_type == 'c':
        p = (S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put'or option_type == 'P' or option_type == 'p':
        p = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0))
    else:
        print("option_type 类型输入错误")
        return None
    return p

# bs公式求期权预估收益
def vanilla_option_return(S, K, T, r, sigma, c, commission=0, q=0, trade_type='buy', option_type='call'):
    """
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: standard deviation of price of underlying asset
    c: 期权初始价格
    commission: 手续费率,如0.005
    q: dividend rate
    trade_type:交易方式
    option_type:期权类型
    return:bs预估收益
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)+episode)
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)+episode)
    if trade_type == 'buy':
        c = c * (1 - commission)
        if option_type == 'call':
            p = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) - c
        elif option_type == 'put':
            p = K * np.exp(-r * T) * norm.cdf(-d2) - S* np.exp(-q * T) * norm.cdf(-d1) - c
        else:
            print("option_type 类型输入错误")
            return None
    elif trade_type == 'sell':
        if option_type == 'call':
            p = c - (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) * (1 - commission)
        elif option_type == 'put':
            p = c - (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)) * (1 - commission)
        else:
            print("option_type 类型输入错误")
            return None
    else:
        print("trade_type 类型输入错误")
        return None
    return p

# 求期权到期价格
def ytm_option(S,K, T, r, sigma, q=0, option_type='call'):
    if option_type == 'call':
        p = np.maximum(S-K, 0)
    elif option_type == 'put':
        p = np.maximum(K-S, 0)
    else:
        print("option_type 类型输入错误")
        return None
    return p


# 求期权到期收益率
def ytm_option_return(S, K, c, commission, trade_type='buy', option_type='call'):
    """
    S: spot price
    K: strike price
    c: 期权初始价格
    trade_type:交易方式
    option_type:期权类型
    commission: 手续费率,如0.005
    return:到期收益
    """

    if trade_type == 'buy':
        c = c * (1 + commission)
        if option_type == 'call':
            p = np.maximum(S-K-c, -c)
        elif option_type == 'put':
            p = np.maximum(K-S-c, -c)
        else:
            print("option_type 类型输入错误")
            return None
    elif trade_type == 'sell':
        if option_type == 'call':
            p = np.maximum(c+(K-S)(1 + commission), c)
        elif option_type == 'put':
            p = np.maximum(c+(S-K)(1 + commission), c)
        else:
            print("option_type 类型输入错误")
            return None
    else:
        print("trade_type 类型输入错误")
        return None
    return p
# 二分法求隐含波动率
def iv_option(S, K, T, r, p, q=0, option_type='call'):
    p_est = 0  # 期权价格估计值
    top = 3  # 波动率上限
    floor = 0  # 波动率下限
    sigma = (floor + top) / 2  # 波动率初始值
    count = 0  # 计数器
    while abs(p - p_est) > 0.0001:
        p_est = vanilla_option(S, K, T, r, sigma, q, option_type)
        # 根据价格判断波动率是被低估还是高估，并对波动率做修正
        count += 1
        if count > 10000:  # 时间价值为0的期权是算不出隐含波动率的，因此迭代到一定次数就不再迭代了
            # sigma = 0
            break

        if p - p_est > 0:  # f(x)>0
            floor = sigma
            sigma = (sigma + top) / 2
        else:
            top = sigma
            sigma = (sigma + floor) / 2
    return sigma


def delta_option(S, K, sigma, r, T, option_type='call'):
    #     计算欧式期权的Delta值
    #     S 基础资产价格
    #     K 行权价
    #     sigma 基础资产价格百分比变化的波动率
    #     r 无风险收益率
    #     T 期权剩余时间（年）
    #     option_type 期权类型；'call'看涨，'put'看跌
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':  # call
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    return delta


def delta_without_sigma(S, K, p, r, T, option_type='call'):
    #     计算欧式期权的Delta值
    #     S 基础资产价格
    #     K 行权价
    #     p 期权价格
    #     r 无风险收益率
    #     T 期权剩余时间（年）
    #     option_type 期权类型；'call'看涨，'put'看跌
    sigma = iv_option(S, K, T, r, p, q=0, option_type=option_type)
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':  # call
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    return delta


def gamma_option(S, K, sigma, r, T):
    #     计算欧式期权的Gamma值
    #     S 基础资产价格
    #     K 行权价
    #     sigma 基础资产价格百分比变化的波动率
    #     r 无风险收益率
    #     T 期权剩余时间（年）
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-pow(d1, 2) / 2) / (S * sigma * np.sqrt(2 * np.pi * T))

def gamma_without_sigma(S, K, p, r, T, option_type='call'):
    """
    :param S: 基础资产价格
    :param K: 行权价
    :param p: 期权价格
    :param r: 无风险收益率
    :param T: 期权剩余时间（年）
    :param option_type: 期权类型；'call'看涨，'put'看跌
    :return: 欧式期权的Gamma值
    """

    sigma = iv_option(S, K, T, r, p, q=0, option_type=option_type)
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-pow(d1, 2) / 2) / (S * sigma * np.sqrt(2 * np.pi * T))

def vega_option(S, K, sigma, r, T):
    #     计算欧式期权的Vega值
    #     S 基础资产价格
    #     K 行权价
    #     sigma 基础资产价格百分比变化的波动率
    #     r 无风险收益率
    #     T 期权剩余时间（年）
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * np.exp(-pow(d1, 2) / 2) / np.sqrt(2 * np.pi)


def theta_option(S, K, sigma, r, T, option_type='call', year=365):
    #     计算欧式期权的Theta值
    #     S 基础资产价格
    #     K 行权价
    #     sigma 基础资产价格百分比变化的波动率
    #     r 无风险收益率
    #     T 期权剩余时间（年）
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return (-(S * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * T)) - r * K * np.exp(
            -r * T) * norm.cdf(d2)) / year
    else:
        return (-(S * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * T)) + r * K * np.exp(
            -r * T) * norm.cdf(-d2)) / year



def btc_vol_df(btc_path='data/BTC-USDT.pkl'):
    btc_price = pd.read_feather(btc_path)
    btc_price['btc_volatility'] = btc_price['open'].pct_change(1).rolling(15 * 24).std()
    return btc_price

def plot_btc(btc_path='data/BTC-USDT.pkl'):
    btc_price = btc_vol_df(btc_path)
    # 创建收益曲线图
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        specs=[
            [{"type": "xy", "secondary_y": True}],
        ],
    )

    fig.add_trace(go.Scatter(x=btc_price['candle_begin_time'],
                             y=btc_price['close'],
                             mode='lines', name='btc_price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=btc_price['candle_begin_time'], y=btc_price['btc_volatility'],
                             mode='lines', name='btc_volatility'), secondary_y=True, row=1, col=1)

    # 更新布局
    fig.update_layout(
        title=f'btc',
        xaxis_title='时间',
        yaxis_title='累积净值',
        legend=dict(x=0, y=1.2, orientation='h')
    )
    import plotly.io as pio

    pio.renderers.default = 'browser'  # 或尝试其他渲染模式
    fig.show()



if __name__ == '__main__':
    # print(vanilla_option(2.611, 2.65, 2, 0.03, 0.18))
    import pandas as pd
    #
    # K_values = range(93000, 98001, 500)
    # call_price_values = [1508,0,348.8,66,18.8,9.43,9.43,9.43,9.43,9.43,9.43]
    # put_price_values = [9.43,9.43, 47.14,330,896,1320,1886,2357,2875,0,3865]
    # values = zip(K_values,call_price_values, put_price_values)
    # results = []
    #
    # for K, cp, pp in values:
    #     call_iv = iv_option(S=94278, K=K, T=1/365/24, r=0.1, p=cp, option_type='call')
    #     put_iv = iv_option(S=94278, K=K, T=1/365/24, r=0.1, p=pp, option_type='put')
    #     results.append({'看涨期权iv': call_iv,'看涨期权价格': cp, '行权价': K, '看跌期权iv': put_iv, '看跌期权价格': pp})
    #
    # results_df = pd.DataFrame(results)
    # print(results_df.to_markdown())
    #

    from datetime import datetime

    K_values = list(range(85000, 96001, 1000)) + [98000,100000, 102000,104000, 105000, 106000, 108000, 110000,112000, 114000, 115000]
    call_price_values = [12825, 12118, 11410, 10750, 10090, 9430, 8817, 8298, 7780, 7261, 6790, 6365, 5563, 4856, 4244, 3678,
                         3442, 3206, 2829, 2452, 2122, 1839, 1744]
    put_price_values = [2640, 2876, 3206, 3536, 3866, 4291, 4668, 5139, 5611, 6035, 6648, 7167, 8440, 9713, 11221,
                        12683, 13438, 14239, 15795, 17398, 19096, 20793, 21642]

    values = zip(K_values, call_price_values, put_price_values)
    # values =[[85000, 12825, 2640],
    #          [86000, 12118, 2876],
    #          [87000, 12118, 2876],
    #          [88000, 12118, 2876],
    #          [89000, 12118, 2876],
    #          [90000, 12118, 2876],
    #          [86000, 12118, 2876],
    #          [86000, 12118, 2876],
    #          [86000, 12118, 2876],
    #          [86000, 12118, 2876],
    #          [86000, 12118, 2876],
    #          [86000, 12118, 2876],
    #           ]
    results = []

    # # 计算到期时长T，以小时为单位
    current_time = datetime(2024, 12, 28, 15, 0)
    expiration_time = datetime(2025, 1, 31, 16, 0)
    T = (expiration_time - current_time).total_seconds() / 3600  # 转换为小时

    for K, cp, pp in values:
        call_iv = iv_option(S=94278, K=K, T=T / 365 / 24, r=0.1, p=cp, option_type='call')
        put_iv = iv_option(S=94278, K=K, T=T / 365 / 24, r=0.1, p=pp, option_type='put')
        results.append(
            {'看涨期权iv': call_iv, '看涨期权价格': cp, '行权价': K, '看跌期权iv': put_iv, '看跌期权价格': pp})

    results_df = pd.DataFrame(results)
    print(results_df.to_markdown())


    # print(iv_option(S=94278, K=86000, T=T / 365 / 24, r=0.1, p=2876, option_type='put'))

    # print(vanilla_option(S=94278, K=86000, T=T / 365 / 24, r=0.1, sigma = 0.589 , option_type='put'))