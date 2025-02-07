# -*- coding:utf-8 -*-
"""
@FileName：pre_symbol_data.py
@Description：
@Author：fdguuwzj
@Time：2025/2/6 11:39
"""
import os
import pandas as pd
from tqdm import tqdm

from config import BACKTEST_SYMBOL_DIR, BACKTEST_DIR
# 使用plotly绘图
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def process_single_symbol(filename, data_path):
    """
    处理单个币种的数据
    Args:
        filename: 文件名
        data_path: 数据路径
    Returns:
        tuple: (symbol, processed_df)
    """
    # 从文件名中提取币种名称,转换为大写
    symbol = filename.split('_')[-1][:-4].upper()

    # 读取文件
    file_path = os.path.join(data_path, filename)
    df = pd.read_pickle(file_path)

    # 去重并排序
    df = df.drop_duplicates()
    df = df.sort_values('hour')

    # 检查数据完整性
    required_columns = ['hour', 'strike_price', 'expiration', 'underlying_price', 'mark_iv']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{symbol} 数据缺少必要列: {missing_cols}")

    # 数据清洗
    df = df.dropna(subset=required_columns)
    df['hour'] = pd.to_datetime(df['hour'])
    df['expiration'] = pd.to_datetime(df['expiration'])

    return symbol, df


def process_symbol_data():
    """
    逐个处理不同币种的期权数据并保存
    """
    # 使用绝对路径
    input_path = os.path.join( '..', BACKTEST_DIR, 'processed_symbol_options')
    output_path = os.path.join('..', BACKTEST_DIR, 'concat_symbol_options')

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 指定要处理的币种
    target_symbols = ['BTC', 'XRP', 'ETH', 'SOL']

    # 创建一个字典来存储每个币种的所有数据
    symbol_data = {symbol: [] for symbol in target_symbols}

    # 遍历数据文件夹,找出包含目标币种的文件
    for filename in tqdm(os.listdir(input_path), desc="处理数据文件"):
        if filename.endswith('.pkl'):
            for symbol in target_symbols:
                if symbol.lower() in filename.lower():
                    try:
                        _, df = process_single_symbol(filename, input_path)
                        symbol_data[symbol].append(df)
                    except Exception as e:
                        print(f"处理 {filename} 时出错: {str(e)}")
                    break

    # 合并并保存每个币种的数据
    for symbol in tqdm(target_symbols, desc="合并币种数据"):
        if not symbol_data[symbol]:
            print(f"警告: 未找到 {symbol} 的任何数据文件")
            continue

        try:
            # 合并该币种的所有数据
            combined_df = pd.concat(symbol_data[symbol], ignore_index=True)

            # 按时间去重
            combined_df = combined_df.sort_values('hour')
            combined_df = combined_df.drop_duplicates(keep='last')

            # 保存处理后的数据
            output_file = os.path.join(output_path, f'{symbol}_data.pkl')
            combined_df.to_pickle(output_file)

            print(f"{symbol} 数据处理完成, 最终数据量: {len(combined_df)}")
            print(f"数据时间范围: {combined_df['hour'].min()} 到 {combined_df['hour'].max()}")

        except Exception as e:
            print(f"合并保存 {symbol} 数据时出错: {str(e)}")
            continue


if __name__ == '__main__':


    # process_symbol_data()
    for symbol in tqdm(['BTC', 'XRP', 'SOL', 'ETH'], desc="计算IV指数"):
        data = pd.read_pickle(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_data.pkl'))
        data['hour'] = pd.to_datetime(data['hour'])
        data = data.sort_values(by='hour')
        # data_19_20 = data[(data['hour'] >= '2019-01-01') & (data['hour'] < '2021-01-01')]
        # data_19_20.to_pickle(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_data_19_20.pkl'))
        # del data_19_20
        # data_21_22 = data[(data['hour'] >= '2021-01-01') & (data['hour'] < '2023-01-01')]
        # data_21_22.to_pickle(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_data_21_22.pkl'))
        # del data_21_22
        # data_23_24 = data[data['hour'] >= '2023-01-01']
        # data_23_24.to_pickle(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_data_23_24_25.pkl'))
        # del data_23_24
        # 保存分段数据




        # 筛选周期权和双周期权
        data['days_to_expiry'] = (pd.to_datetime(data['expiration']) - data['hour']).dt.days
        weekly_options = data[data['days_to_expiry'].between(3, 7, inclusive='right')]  # 周期权
        biweekly_options = data[data['days_to_expiry'].between(7, 14, inclusive='right')]  # 双周期权

        # 计算平值期权
        weekly_atm = weekly_options.groupby('hour').apply(
            lambda x: x.iloc[(x['strike_price'] - x['underlying_price']).abs().argsort()[:2]]
        ).reset_index(drop=True)

        biweekly_atm = biweekly_options.groupby('hour').apply(
            lambda x: x.iloc[(x['strike_price'] - x['underlying_price']).abs().argsort()[:2]]
        ).reset_index(drop=True)


        # 计算iv指数 - 使用mark_iv的加权平均
        def calc_iv_index(weekly, biweekly):
            w1 = biweekly['days_to_expiry'].mean() / (
                        biweekly['days_to_expiry'].mean() - weekly['days_to_expiry'].mean())
            w2 = 1 - w1
            return w1 * weekly['mark_iv'].mean() + w2 * biweekly['mark_iv'].mean()


        iv_index = pd.DataFrame()
        iv_index['hour'] = sorted(set(weekly_atm['hour']).intersection(set(biweekly_atm['hour'])))
        iv_index['iv'] = iv_index['hour'].apply(lambda h: calc_iv_index(
            weekly_atm[weekly_atm['hour'] == h],
            biweekly_atm[biweekly_atm['hour'] == h]
        ))
        iv_index.to_csv(f'{symbol}_iv_index.csv', index=False)
        # 计算一年时间窗口的滚动历史分位数
        window_size = '365D'  # 一年的时间窗口
        iv_index['iv_percentile_1year'] = iv_index.set_index('hour')['iv'].rolling(window_size, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        ).reset_index()['iv']
        # 计算历史分位数
        iv_index['iv_percentile'] = iv_index['iv'].rank(pct=True)

        # 创建子图
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=(f'{symbol} IV指数', f'{symbol} IV历史分位数'),
                            shared_xaxes=True)  # 共享x轴

        # IV指数图
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(iv_index['hour']),
                       y=iv_index['iv'],
                       mode='lines',
                       name='IV指数'),
            row=1, col=1
        )

        # IV历史分位数图
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(iv_index['hour']),
                       y=iv_index['iv_percentile'],
                       mode='lines',
                       name='IV历史分位数'),
            row=2, col=1
        )
        # IV历史分位数图
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(iv_index['hour']),
                       y=iv_index['iv_percentile_1year'],
                       mode='lines',
                       name='IV历史分位数(1year)'),
            row=2, col=1
        )

        # 更新布局
        fig.update_layout(
            height=800,
            title_text=f"{symbol}期权IV指数分析",
            showlegend=True
        )

        # 保存图表为HTML文件
        fig.write_html(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_iv_analysis.html'))

        # 保存带有分位数的数据
        iv_index.to_csv(os.path.join(BACKTEST_SYMBOL_DIR, f'{symbol}_iv_index_with_percentile.csv'), index=False)
