# -*- coding:utf-8 -*-
"""
@FileName：market_api.py
@Description：
@Author：fdguuwzj
@Time：2025-02-27 19:56
"""
import os
import time
import urllib
from typing import Tuple, Dict, Any

import httpx
import pandas as pd
import orjson
from loguru import logger
from pandas import Timestamp
from pandas._libs import NaTType


endpoint = 'https://www.deribit.com/api/v2'

def _http_build_query_string(params_dict: Dict) -> str:
    if not params_dict:
        return ''
    param_str = '&'.join(f"{urllib.parse.quote_plus(str(k))}={urllib.parse.quote_plus(str(v))}" for k, v in sorted(params_dict.items()))
    return param_str

def _deribit_http_request(func_name: str, method: str, url_path: str, params: Dict = None, timestamp: object = True,
                          ac: object = None,
                          api_key: str = None,
                          api_secret: str = None,
                          exclude_error: str = None,
                          proxy: str = None,
                          http_json:dict = None,
                          **kwargs: object) -> Tuple[Any, Dict]:

    params = params or {}
    if ac:
        api_key = ac.key
        api_secret = ac.secret


    if api_key and api_secret:
        account_name = ac.name.split('/')[0]
        signature = _deribit_get_access_token(endpoint=endpoint, account_name=account_name, api_key=api_key, secret=api_secret, proxy=proxy)
        headers = {
            'Authorization': f'Bearer {signature}',
        }
    else:
        headers = {}
    url = f'{endpoint}{url_path}'



    try:
        rs = retry_http_request(func_name=func_name, method=method, url=url, params=params, headers=headers, retry_params=params, exclude_error=exclude_error, ac=ac, proxy=proxy,http_json=http_json)
        rs = rs.json()
        retMsg = rs.get('retMsg', None)
        if retMsg:
            if 'API key' in retMsg or 'IP address' in retMsg:
                return None, {'error': retMsg}
    except Exception as e:
        return None, {'error': e}
    return rs, {}


def _deribit_auth(endpoint: str, api_key: str, secret: str, grant_type: str = 'client_credentials', proxy: str = None) -> \
tuple[Any, Timestamp]:
    url = f'{endpoint}/public/auth?client_id={api_key}&client_secret={secret}&grant_type={grant_type}'
    update_time = pd.Timestamp.utcnow().tz_localize(None)
    response = retry_http_request(func_name='_deribit_auth', url=url, method='GET', proxy=proxy)
    access_token = response.json()['result']['access_token']
    return access_token, update_time


def _load_account_token_cache(account_name: str) -> tuple[Any, Timestamp | NaTType] | tuple[None, None]:
    dir_path = 'token/deribit'
    file_path = f'{dir_path}/{account_name}.json'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())
            return data['access_token'], pd.Timestamp(data['update_time'])
    return None, None


def _save_account_token_cache(account_name: str, access_token_cache: str, update_time_cache: pd.Timestamp):
    dir_path = 'token/deribit'
    file_path = f'{dir_path}/{account_name}.json'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'wb') as f:
        data = {'access_token': access_token_cache, 'update_time': pd.Timestamp(update_time_cache).tz_localize(None).isoformat()}
        f.write(orjson.dumps(data))


def _deribit_get_access_token(endpoint: str, account_name: str, api_key: str, secret: str, grant_type: str = 'client_credentials', proxy: str = None) -> str:
    current_time = pd.Timestamp.utcnow().tz_localize(None)
    _access_token_cache, _update_time_cache = _load_account_token_cache(account_name)
    if not _access_token_cache or (current_time - _update_time_cache) > pd.Timedelta('850s'):
        _access_token_cache, _update_time_cache = _deribit_auth(endpoint=endpoint, api_key=api_key, secret=secret, grant_type=grant_type, proxy=proxy)
        _save_account_token_cache(account_name=account_name, access_token_cache=_access_token_cache, update_time_cache=_update_time_cache)
    return _access_token_cache


def get_ticker_price(symbol):
    url = f'{endpoint}/public/ticker?instrument_name={symbol}'
    response = retry_http_request(func_name='get_ticker_price', url=url, method='GET')
    return response.json()['result']['last_price']

def make_order(instrument_name, side, order_type , contracts: int, ac, price: float = None, proxy:str=None):
    url = f'/private/{side}'
    if order_type == 'limit':
        params = {
            'instrument_name': instrument_name,
            'contracts': str(contracts),
            'price': float(price),
            # 'post_only': True,
            'label': 'hedging'
        }
    elif order_type == 'market':
        params = {
            'instrument_name': instrument_name,
            'contracts': str(contracts),
            'type': order_type,
            # 'post_only': True,
            'label': 'hedging'
        }
    response, error = _deribit_http_request(func_name='make_order', method='POST', url_path=url, http_json={'method': 'private/buy','params':params}, ac=ac, proxy=proxy)
    return response, error


def retry_http_request(func_name: str, url: str, method: str = 'GET', params=None, content=None, data=None, http_json=None, headers: Dict = None, timeout: float = 6, retry_times: int = 5, sleep_seconds: float = 2,  proxy: str = None, **kwargs) -> httpx.Response | None:
    """
    try to run retry_times times, sleep retry_sleep_seconds between each attempt, to reduce duplicate log, only log.error for the last attempt
    """
    for attempt in range(retry_times):
        try:
            with httpx.Client(proxy=proxy) as client:
                response = client.request(method=method, url=url, params=params, content=content, data=data, json=http_json, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response
        except (httpx.NetworkError, httpx.RequestError, httpx.TimeoutException, TimeoutError) as e:
            attempt += 1
            err = e.args[0] if e.args else e.args
            # The first try not log error
            error = f'retry_http_request network retry, func={func_name} kwargs={kwargs} sleep_seconds={sleep_seconds} attempt={attempt}/{retry_times} \nerror: {err} '
            if attempt == retry_times:
                logger.error(error)


if __name__ == '__main__':
    print(get_ticker_price('BTC-PERPETUAL'))