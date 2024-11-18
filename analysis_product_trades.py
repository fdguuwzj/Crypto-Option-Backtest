# -*- coding:utf-8 -*-
"""
@FileName：analysis_product_trades.py
@Description：
@Author：fdguuwzj
@Time：2024/11/15 16:16
"""
import os

import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from config import PRODUCT_LOG_DIR
log_filename = "transaction_log-355249-BTC-10_31_2021, 4_00_00 PM_11_19_2024, 8_22_28 AM.csv"
trades = pd.read_csv(os.path.join(PRODUCT_LOG_DIR, log_filename))
filtered_trades = trades[trades['Instrument'].notna()]
filtered_trades['Date'] = pd.to_datetime(filtered_trades['Date'])
filtered_trades = filtered_trades.sort_values('Date')
# 去掉 btc spot 交易
filtered_trades = filtered_trades[filtered_trades['Instrument'] != 'BTC_USDT']
filtered_trades['quote_volume'] = filtered_trades['Cash Flow']*filtered_trades['Index Price']

filtered_trades['accumulate_btc_pnl'] = filtered_trades['Cash Flow'].cumsum()
filtered_trades['accumulate_pnl'] = filtered_trades['accumulate_btc_pnl'] * filtered_trades['Index Price']
filtered_trades = filtered_trades[['Date', 'Instrument','Type','Side' ,'Index Price', 'Price', 'Amount', 'quote_volume', 'accumulate_btc_pnl', 'accumulate_pnl']]
filtered_trades.to_csv(os.path.join(PRODUCT_LOG_DIR, f'processed_{log_filename}'), index=False)
# 创建收益曲线图
fig = make_subplots(
    rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    specs=[
        [{"type": "xy", "secondary_y": True}],
        # [{"type": "xy", "secondary_y": False}],
        # [{"type": "xy", "secondary_y": False}],
        # [{"type": "table"}],
    ],
)
# 添加累积净值折线图
fig.add_trace(go.Scatter(x=filtered_trades['Date'], y=filtered_trades['accumulate_pnl'],
                         mode='lines+markers', name='累积净值'), row=1, col=1)


# # 调整每笔收益散点图以使其更加显眼，并在每个点上标明此单收益
# fig.add_trace(go.Scatter(x=filtered_trades['current_time'], y=filtered_trades['quote_volume'],
#                          mode='markers+text', name='每笔收益', marker_color=filtered_trades['pnl_type'].map({'premium': 'green', 'return': 'red'}), opacity=1,),
#                          row=2, col=1)
#


# 更新布局
fig.update_layout(
    title=f'实盘收益曲线图',
    xaxis_title='时间',
    yaxis_title='累积净值',
    yaxis2_title='最大回撤',
    legend=dict(x=0, y=1.2, orientation='h')
)
import plotly.io as pio

pio.renderers.default = 'browser'  # 或尝试其他渲染模式

# pio.write_html(fig, f"{self.save_dir}/{self.strategy}收益曲线图_exe_price_gear={self.exe_price_gear}_mature_gear={self.mature_gear}.html")

fig.show()
print(filtered_trades)