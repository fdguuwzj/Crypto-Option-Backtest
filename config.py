# -*- coding:utf-8 -*-
"""
@FileName：config.py
@Description：
@Author：fdguuwzj
@Time：2024/11/18 16:27
"""
import os

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
BACKTEST_DIR = os.path.join(DATA_DIR, 'backtest_data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
PRODUCT_LOG_DIR = os.path.join(DATA_DIR, 'product_logs')
BACKTEST_SYMBOL_DIR = os.path.join(DATA_DIR,  'backtest_data', 'concat_symbol_options')

UNRISK_RATE = 0.1
EPSILON = 1e-16

SNAPSHOT_TIME = 'snapshot_time'
EXPIRATION = 'expiration'
EXE_PRICE = 'strike_price' # 'exe_price'
CALL = 'call'
PUT = 'put'
OPTION_NAME = 'instrument_name'
MARK_PRICE = 'mark_price'
BID_PRICE = 'bid_price'
ASK_PRICE = 'ask_price'
TYPE = 'type'