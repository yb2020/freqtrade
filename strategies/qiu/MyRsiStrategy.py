# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
# from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class MyRsiStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }
    minimal_roi = {}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.02

    # Trailing stoploss - 我们使用自定义动态止损,因此必须禁用追踪止损
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Custom stoploss
    use_custom_stoploss = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")  # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                },
            },
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe["plus_dm"] = ta.PLUS_DM(dataframe)
        # dataframe["plus_di"] = ta.PLUS_DI(dataframe)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe["rsi"] - 50)
        # dataframe["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # MACD
        # macd = ta.MACD(dataframe)
        # dataframe["macd"] = macd["macd"]
        # dataframe["macdsignal"] = macd["macdsignal"]
        # dataframe["macdhist"] = macd["macdhist"]

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        # --- RSI底背离入场逻辑 ---

        # 1. 定义价格和RSI在过去周期内的最低值
        # 延长回溯周期以捕捉更长时间的真实底部
        dataframe["price_low_10"] = dataframe["low"].shift(1).rolling(10).min()  # 前10根K线最低价
        dataframe["rsi_low_10"] = dataframe["rsi"].shift(1).rolling(10).min()  # 前10根K线最低RSI

        # 2. 定义"潜在反转信号": RSI底背离(多路径识别)

        # 路径1: 强背离[严格条件]
        strong_divergence = (
            (dataframe["rsi"] < 30)  # RSI 超卖
            & (dataframe["low"] < dataframe["price_low_10"] * 0.998)  # 价格突破更明显
            & (dataframe["rsi"] > dataframe["rsi_low_10"] + 5)  # RSI 明显上升 (+5)
            & (dataframe["rsi"] > dataframe["rsi"].shift(1) + 2)  # 且加速上升 (+2)
            & (dataframe["volume"] > dataframe["volume"].rolling(10).mean() * 1.2)  # 加入成交量确认
        )

        # 路径2: 弱背离[放宽 RSI 条件]
        weak_divergence = (
            (dataframe["rsi"] < 30)  # RSI 超卖
            & (dataframe["low"] < dataframe["price_low_10"])  # 价格创新低
            & (dataframe["rsi"] > dataframe["rsi_low_10"] + 3)  # 弱背离 (+3 即可)
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))  # RSI 上升[不要求加速]
        )

        # 最终信号: 两条路径任一满足
        potential_reversal_signal = strong_divergence | weak_divergence

        # 将信号向前“传播”3根K线,形成一个有效的“信号窗口”
        dataframe["signal_window"] = potential_reversal_signal.rolling(
            window=3, min_periods=1
        ).max()

        # 3. 定义完整的"价格行为确认信号"
        # 为路径2[弱背离]增加更强的价格确认要求

        # 基础价格确认[适用于路径1强背离]
        basic_price_confirmation = (
            (dataframe["close"] > dataframe["price_low_10"])  # 价格收复失地
            & (dataframe["close"] > dataframe["open"])  # K线为实体阳线
        )

        # 强价格确认[适用于路径2弱背离]
        strong_price_confirmation = (
            (dataframe["close"] > dataframe["price_low_10"] * 1.005)  # 价格突破 0.5%
            & (dataframe["close"] > dataframe["open"])  # K线为实体阳线
            & (
                (dataframe["close"] - dataframe["open"]) / dataframe["open"] > 0.005
            )  # 阳线实体 > 0.5%
            & (dataframe["volume"] > dataframe["volume"].rolling(20).mean() * 1.2)  # 成交量放大 20%
        )

        # 判断是哪条路径触发的信号[用于价格确认选择]
        is_strong_div_shifted = (
            (dataframe["rsi"].shift(1) < 30)
            & (dataframe["low"].shift(1) < dataframe["price_low_10"].shift(1))
            & (dataframe["rsi"].shift(1) > dataframe["rsi_low_10"].shift(1) + 5)
            & (dataframe["rsi"].shift(1) > dataframe["rsi"].shift(2) + 2)
        )

        # 根据路径选择确认条件
        price_confirmation = np.where(
            is_strong_div_shifted,
            basic_price_confirmation,  # 强背离用基础确认
            strong_price_confirmation,  # 弱背离用强确认
        )

        # 4. 最终入场条件: 在信号窗口内,等待价格行为的最终确认
        conditions = (
            (dataframe["signal_window"].shift(1) == 1)  # 条件1: 处于激活的信号窗口内
            & price_confirmation  # 条件2: 出现完整的价格行为确认信号
            & (dataframe["volume"] > 0)
        )

        # 5. 开仓,同时标记其他量化分析标记
        # 区分强背离和弱背离的入场信号
        strong_entry = conditions & is_strong_div_shifted
        weak_entry = conditions & ~is_strong_div_shifted

        dataframe.loc[strong_entry, "enter_long"] = 1
        dataframe.loc[strong_entry, "enter_tag"] = "buy_strong"
        dataframe.loc[weak_entry, "enter_long"] = 1
        dataframe.loc[weak_entry, "enter_tag"] = "buy_weak"

        # 将止损价格存储在专用列中供 custom_stoploss 使用
        # 放宽止损距离以给予更多空间
        dataframe.loc[conditions, "stop_price"] = dataframe["price_low_10"] * 0.995

        return dataframe

    # 复写出场逻辑
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # dataframe.loc[
        #     (
        #         (
        #             qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
        #         )  # Signal: RSI crosses above sell_rsi
        #         & (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #     ),
        #     "exit_long",
        # ] = 1
        return dataframe

    # 自定义止损
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        动态止损: 基于入场时的 price_low_5
        止损位 = price_low_5 * 0.998
        """
        # 从 dataframe 中获取止损价格
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # 找到入场时的止损价格
        if len(dataframe) > 0:
            # 使用入场时间找到对应的 K 线
            entry_candle = dataframe[dataframe["date"] <= trade.open_date_utc].iloc[-1]
            if "stop_price" in entry_candle and not pd.isna(entry_candle["stop_price"]):
                stop_price = entry_candle["stop_price"]
                # 计算相对于当前价格的止损百分比
                stop_loss_pct = (stop_price - current_rate) / current_rate
                return stop_loss_pct

        # 如果无法获取,使用默认止损
        return None

    # 自定义出场
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        分类出场策略: 根据 enter_tag 设置不同的出场条件
        - 强背离(buy_strong): RSI > 70 (更贪婪)
        - 弱背离(buy_weak): RSI > 60 (更保守)
        """
        # 获取当前这笔交易的入场标签
        enter_tag = trade.enter_tag or ""

        # 获取最新的RSI值
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        # 只在有盈利时考虑出场
        if current_profit <= 0:
            return None

        # 如果是 'buy_strong',则 RSI 上穿 70 才出场
        if enter_tag == "buy_strong":
            if qtpylib.crossed_above(dataframe["rsi"], 70).iloc[-1]:
                return "exit_strong_rsi_70"

        # 如果是 'buy_weak',则 RSI 上穿 60 就出场
        if enter_tag == "buy_weak":
            if qtpylib.crossed_above(dataframe["rsi"], 60).iloc[-1]:
                return "exit_weak_rsi_60"

        # 其他情况不出场
        return None
