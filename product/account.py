# -*- coding:utf-8 -*-
"""
@FileName：account.py
@Description：
@Author：fdguuwzj
@Time：2025-03-03 16:44
"""
import pandas as pd

from product.api.market_api import _deribit_http_request


class Account:
    def __init__(self, key, secret, proxy, name, target:str='BTC'):
        self.key = key
        self.secret = secret
        self.name = name
        self.proxy = proxy
        self.asset = 0
        self.pos_delta = 0
        self.option_delta = 0
        self.future_delta = 0
        self.current_position:pd.DataFrame = None
        self.equity = None
        self.target = target

    def __repr__(self):
        return f"""{'-' * 32}
#   {self.name}信息如下：
#   资产：{self.asset:.2f} {self.target}
#   权益：{self.equity:.2f} usd
#   期权delta：{self.option_delta:.2f} {self.target}
#   期货delta：{self.future_delta:.2f} {self.target}
#   总delta：{self.pos_delta:.2f} {self.target}
{'-' * 32}
"""
#   账户持仓：{self.current_position.to_markdown(tablefmt='grid')}


    def update_position(self):
        rs, err = _deribit_http_request(ac=self, func_name='get_account_summaries', method='GET',
                                        url_path='/private/get_account_summaries', params={}, proxy=self.proxy)
        currency = [c for c in rs['result']['summaries'] if c['currency'] == 'BTC'][0]
        self.asset = currency['equity']
        self.equity = currency['total_equity_usd']
        rs, err = _deribit_http_request(ac=self, func_name='get_positions', method='GET',
                                        url_path='/private/get_positions', params={}, proxy=self.proxy)
        self.option_delta = sum([c['delta'] for c in rs['result'] if c['kind'] == 'option'])
        self.future_delta = sum([c['delta'] for c in rs['result'] if c['kind'] == 'future'])
        self.pos_delta = self.future_delta + self.asset + self.option_delta





