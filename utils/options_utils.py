# -*- coding:utf-8 -*-
"""
@FileName：options_utils.py
@Description：fdguuwzj self-made option utils toolkit
@Author：fdguuwzj
@Time：2023-12-11 16:39
"""
import numpy as np
from scipy.stats import norm

eps=1e-10
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
    """
    episode = 0.00000001
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + episode)
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + episode)
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)+eps)
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)+eps)
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
    top = 1  # 波动率上限
    floor = 0  # 波动率下限
    sigma = (floor + top) / 2  # 波动率初始值
    count = 0  # 计数器
    while abs(p - p_est) > 0.0001:
        p_est = vanilla_option(S, K, T, r, sigma, q, option_type) / 100
        # 根据价格判断波动率是被低估还是高估，并对波动率做修正
        count += 1
        if count > 1000:  # 时间价值为0的期权是算不出隐含波动率的，因此迭代到一定次数就不再迭代了
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
    #     sigma 基础资产价格百分比变化的波动率
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


if __name__ == '__main__':
    print(vanilla_option(2.611, 2.65, 2, 0.03, 0.18))
    # print(1)
