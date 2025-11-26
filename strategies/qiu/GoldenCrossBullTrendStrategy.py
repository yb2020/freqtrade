from functools import reduce

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy


class GoldenCrossBullTrendStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # --- 核心配置 ---
    timeframe = "1h"
    can_short = False

    # 1. 止损: 你要求的底线
    stoploss = -0.035

    # 2. 移动止损: 这是核心!
    # 这里的逻辑是: 只要赚到 2.5%, 就把止损拉到 0.5% (保本微赚)
    # 然后就一直拿, 直到行情回头打掉止损线
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.025
    trailing_stop_positive = 0.005

    # 3. 止盈 ROI: 彻底删除时间限制!
    # 之前亏损是因为 120分钟、1440分钟 强制保本离场, 导致大行情没吃到
    # 现在改为: 要么赚 10% 以上, 要么就被移动止损踢出局, 绝不主动小赚离场
    minimal_roi = {
        "0": 0.10,  # 只有赚到 10% 才考虑硬止盈
    }

    # --- 参数 ---
    # RSI 50 金叉是趋势启动最标准的信号
    buy_rsi_min = IntParameter(45, 55, default=50, space="buy")
    # buy_rsi_max = IntParameter(55, 65, default=65, space="buy")

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema50": {"color": "orange"},
                "sma200": {"color": "white"},
            },
            "subplots": {
                "RSI": {"rsi": {"color": "red"}},
                "ADX": {"adx": {"color": "blue"}},
            },
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 只有最基础的指标, 越简单越有效
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["sma200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_ma"] = ta.SMA(dataframe["rsi"], timeperiod=9)
        dataframe["adx"] = ta.ADX(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            # 1. 趋势: 多头排列
            (dataframe["ema50"] > dataframe["sma200"])
            &
            # 2. 强度: 价格在均线之上
            (dataframe["close"] > dataframe["ema50"])
            &
            # 3. 动能: RSI 处于多头区域
            (dataframe["rsi"] > self.buy_rsi_min.value)
            &
            # 4. 触发器: RSI 金叉
            (dataframe["rsi"] > dataframe["rsi_ma"])
            & (dataframe["rsi"].shift(1) <= dataframe["rsi_ma"].shift(1))
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 不设任何主动离场, 让 ROI 和 Trailing Stop 决定生死
        # 趋势止损: 当价格下穿 sma200 长期生命线时, 无条件离场
        conditions = []
        conditions.append(
            (dataframe["close"] < dataframe["sma200"])
            & (dataframe["close"].shift(1) >= dataframe["sma200"].shift(1))
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe
