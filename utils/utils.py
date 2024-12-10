# -*- coding:utf-8 -*-
"""
@FileName：utils.py
@Description：
@Author：fdguuwzj
@Time：2024/12/3 17:12
"""
import time
import traceback

import requests
from loguru import logger
from urllib3.exceptions import ReadTimeoutError


def retry_wrapper(
        func, params=None, func_name='', retry_times=5, sleep_seconds=1, if_exit=True
):
    """
    需要在出错时不断重试的函数，例如和交易所交互，可以使用本函数调用。
    :param func:            需要重试的函数名
    :param params:          参数
    :param func_name:       方法名称
    :param retry_times:     重试次数
    :param sleep_seconds:   报错后的sleep时间
    :param if_exit:         报错是否退出程序
    :return:
    """
    if params is None:
        params = {}
    for _ in range(retry_times):
        try:
            if 'timestamp' in params:

                params['timestamp'] = int(time.time() * 1000) - 2000
            result = func(params=params)
            return result
        except Exception as e:
            logger.error(f'{func_name} 报错，程序暂停{sleep_seconds}(秒)')
            # 出现1021错误码之后，刷新与交易所的时差
            msg = str(e).strip()
            logger.error(msg)
            if 'binance Account has insufficient balance for requested action' in msg:
                logger.warning(f'{func_name} 现货下单资金不足')
                raise ValueError(func_name, '现货下单资金不足')
            elif '-2011' in msg:
                logger.warn(f'{func_name} 无需撤销订单')
                break
            elif '-2021' in msg:
                logger.error(f'{e}')
                exit()
            elif '-2022' in msg:
                logger.warning(f'{func_name} ReduceOnly订单被拒绝, 合约仓位已经平完')
                raise ValueError(func_name, 'ReduceOnly订单被拒绝, 合约仓位已经平完')
            elif '-2019' in msg:
                logger.warn(f'{func_name} 合约下单资金不足')
                raise ValueError(func_name, '合约下单资金不足')
            elif '-2015' in msg and 'Invalid API-key' in msg:
                # {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action, request ip: xxx.xxx.xxx.xxx"}
                logger.error(f'{func_name} API配置错误，可能写错或未配置权限')
                break
            elif '-4164' in msg:
                logger.warning(f'{func_name} 不满足最小下单金额')
                break
            elif 'binance requires "apiKey" credential' in msg:
                logger.error(f'{func_name} 需要准确配置API Key 和 Secret，才能获取账户数据，请检查配置')
                exit(1)
            elif isinstance(e, ReadTimeoutError) :
                logger.error(f'{func_name} 网络超时，程序暂停{sleep_seconds}(秒)')
            else:
                logger.error(f'{e}，报错内容如下')
                logger.error(f'ERR:{func_name}', sep='-')
                logger.debug(traceback.format_exc())
                logger.error(f'ERR:{func_name}', sep='-')
            time.sleep(sleep_seconds)
    else:
        if if_exit:
            raise ValueError(func_name, '报错重试次数超过上限，程序退出。')


def get_binance_usdt_futures_tickers():
    url = "https://fapi.binance.com//eapi/v1/userTrades"  # Binance Futures API endpoint
    response = requests.get(url)
    data = response.json()

    usdt_futures_tickers = [
        symbol['symbol'] for symbol in data['symbols']
        if symbol['quoteAsset'] == 'USDT'
    ]

    return usdt_futures_tickers