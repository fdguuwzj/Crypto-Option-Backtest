# -*- coding:utf-8 -*-
"""
@FileName：delta_hedger.py
@Description：
@Author：fdguuwzj
@Time：2025-02-26 12:26
"""
from math import floor

import numpy as np

import time
from apscheduler.schedulers.background import BackgroundScheduler as Scheduler
from loguru import logger
from zmq.devices import Proxy

from product.account import Account
from product.api.market_api import get_ticker_price, make_order
from product.utils import MessageSender

PROXY ='http://127.0.0.1:10809'
min_qty = {'BTC-PERPETUAL': 0.5}


def apply_precision(price, min_tick):
    change = 1 / min_tick
    return (price * change - (price * change) % (min_tick * change)) / change


class Hedger:
    def __init__(self, accounts: [Account], hedge_fun, target, message_sender: MessageSender, hedge_interval='1h'):
        self.accounts = accounts

        self.hedge_fun = hedge_fun
        self.hedge_interval = hedge_interval
        self.target = target
        self.message_sender = message_sender

    def hedge(self, account: Account, threshold1, threshold2):
        account.update_position()
        threshold1 = threshold1*account.equity
        threshold2 = threshold2*account.equity
        msg = f'now account {account.name}: \nhedge_threshold1 is {threshold1:.2f}\nhedge_threshold2 {threshold2:.2f}'
        logger.info(msg)
        self.message_sender.send(msg)
        if self.hedge_fun == 'default':
            target_price = get_ticker_price(self.target)
            cash_delta = account.pos_delta * target_price
            # now_options_cash_delta = self.account.option_delta * target_price
            if abs(cash_delta) >= threshold1:  # 当delta达到设定的阈值时，返回true
                msg = f'current delta is {cash_delta:.2f} out of {threshold1:.2f}\nstart hedging~'
                logger.info(msg)
                self.message_sender.send(msg)
                # exit()
                adjust_amount = -np.sign(cash_delta) * abs(abs(threshold2) - abs(cash_delta)) / target_price
                # to_amount = adjust_amount + (cash_delta - now_options_cash_delta) / target_price
                side = 'buy' if adjust_amount > 0 else 'sell'

                # msg = f'{account.name} order maked.\n{side} {self.target} {int(abs(adjust_amount) * target_price // 10):.2f} contracts at price: {target_price:.2f} usd'
                # logger.info(msg)
                # self.message_sender.send(msg)
                # exit()
                response, error = make_order(instrument_name=self.target, side=side, order_type='limit',
                           price=apply_precision(target_price, min_qty[self.target]),
                           contracts=int(abs(adjust_amount) * target_price // 10), ac=account, proxy=account.proxy)
                order_msg = response['result']['order']
                order_prcie = order_msg['price']
                msg = f'{account.name} order maked.\n{side} {self.target} {int(abs(adjust_amount) * target_price // 10):.2f} contracts at price: {order_prcie:.2f} usd'
                logger.info(msg)
                self.message_sender.send(msg)
                # logger.info(f'current_position.records: {self.account.current_position.records}')
                # logger.info(f'current_swap_record: {self.account.current_position.records[self.target]}')
                # logger.info(f'current_capital: {self.account.current_position.current_capital}')
            account.update_position()
            # self.trading_logger.trade_positions.append(get_position(self.account.current_position))
    def hedge_accounts(self, threshold1, threshold2):
        for account in self.accounts:
            self.hedge(account, threshold1, threshold2)

    def run(self, **kwargs):
        scheduler = Scheduler()
        # scheduler.add_job(self.hedge_accounts, 'cron', second=0, kwargs=kwargs)
        scheduler.add_job(self.hedge_accounts, 'cron', minute=0, kwargs=kwargs)
        scheduler.start()
        while True:
            time.sleep(1)


if __name__ == '__main__':
    # a = apply_precision(70000.8, 0.5)

    # for test in [
    #     {'cash_delta': 1000, 'threshold1': 500, 'threshold2': 200, 'target_price': 50000, 'expected': -0.016},
    #     {'cash_delta': -1000, 'threshold1': 500, 'threshold2': 200, 'target_price': 50000, 'expected': 0.016},
    #     {'cash_delta': 2000, 'threshold1': 500, 'threshold2': 200, 'target_price': 50000, 'expected': 0},
    #     {'cash_delta': 1500, 'threshold1': 700, 'threshold2': 300, 'target_price': 60000, 'expected': -0.02},
    #     {'cash_delta': -1500, 'threshold1': 700, 'threshold2': 300, 'target_price': 60000, 'expected': 0.02},
    #     {'cash_delta': 2500, 'threshold1': 1000, 'threshold2': 500, 'target_price': 70000, 'expected': -0.014},
    #     {'cash_delta': -2500, 'threshold1': 1000, 'threshold2': 500, 'target_price': 70000, 'expected': 0.014},
    #     {'cash_delta': 3000, 'threshold1': 1200, 'threshold2': 600, 'target_price': 80000, 'expected': -0.015},
    #     {'cash_delta': -3000, 'threshold1': 1200, 'threshold2': 600, 'target_price': 80000, 'expected': 0.015},
    # ]:
    #
    #     adjust_amount = -np.sign(test['cash_delta']) * abs(abs(test['threshold2']) - abs(test['cash_delta'])) / test['target_price']
    #     print(f'now cash_delta is {test["cash_delta"]}， now threshold1 is {test["threshold1"]}， now threshold2 is {test["threshold2"]}， now target_price is {test["target_price"]}：now adjust_amount is {adjust_amount*test["target_price"]}')
    message_sender = MessageSender(key='5426883973', token='7586997310:AAGKmJP2v3w5NGJNfuAECU0aeF64F9LeRRk', platform='Telegram')
    message_sender.send(message='Hello, hedge robot is starting!')
    tt_db_04 = Account(
                      # key='oXxTLdSZ',
                      # secret='XpXa4jMRWov-toZ6yEsU3h1bkkl-m7sE-o4OOelXxjs',
                      key='PzoCWwex',
                      secret='bY213TaRt1bJF71-TzyCbz3F0dEeGl3VM-Ddr9K5KG4',
                      proxy=PROXY,
                      name='tt_db_04',
                      target='BTC'
                      )
    tt_db_05 = Account(
                      # key='oXxTLdSZ',
                      # secret='XpXa4jMRWov-toZ6yEsU3h1bkkl-m7sE-o4OOelXxjs',
                      key='O4_qDU3N',
                      secret='LtKUCQ33qm5ysmgwjB_WpFIBmB4rtKl_yugYTb_tcz4',
                      proxy=PROXY,
                      name='tt_db_05',
                      target='BTC'
                      )
    hedger = Hedger(accounts=[tt_db_04, tt_db_05],
                    hedge_fun = 'default',target='BTC-PERPETUAL',
                    message_sender=message_sender)
    # hedger.hedge(0.5,0.25)
    hedger.run(threshold1=0.5, threshold2 = 0.25)
