"""
# 交易所参数配置，
可以通过get_binance_ex()自动初始化ccxt.binance

```python
from config.binance import get_binance_ex
exchange = get_binance_ex()
```
"""

import ccxt
from trade_analysis.config import bn_api_key, bn_api_secret


from loguru import logger

# ================================================================
# 交易所参数设置
# ================================================================
BINANCE_CONFIG = {
    'timeout': 6000,
    # 'rateLimit': 10,
    # 'verbose': False,
    # 'enableRateLimit': False,
    # 本地的代理设置
    # - 可以使交易所流量都通过科学上网软件，
    # - 比如用clash的话：proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    # - 如果身在国外，或者在服务器运行，无需额外配置
    'proxies': {},
    # 'proxies': {'http': 'http://127.0.0.1:1082', 'https': 'http://127.0.0.1:1082'},
    'options': {
        'adjustForTimeDifference': False,  # solve time recvWindow
        'recvWindow': 5000,  # binance rule: (timestamp < (serverTime + 1000) && (serverTime - timestamp) <= recvWindow)
    }
}


def get_binance_ex(proxies: dict = None, **kwargs) -> ccxt.binance:
    """
    自动生成可以交互的exchange对象，可以使用ccxt中binance的所有功能
    :param proxies: 代理配置
    :param kwargs: 其他binance接受的配置参数
    :return:
    """
    try:
        ex = ccxt.binance({
            'apiKey': bn_api_key,
            'secret': bn_api_secret,
            **BINANCE_CONFIG,
            **kwargs
        })

        if proxies:
            ex.proxies = proxies
        ex.nonce = lambda: ex.milliseconds() - 1000

        return ex
    except Exception as e:
        logger.error(f'{e}')
        logger.error('初始化币安交易所过程中出错，参数如下：')
        logger.debug(str({
            'apiKey': bn_api_key,
            'secret': bn_api_secret,
            **BINANCE_CONFIG,
            'proxies': proxies,
            **kwargs
        }))
        exit(1)
