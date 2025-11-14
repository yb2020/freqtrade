# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.persistence import LocalTrade
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


class SMARSIStrategy(IStrategy):
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
    # minimal_roi = {"60": 0.01, "30": 0.02, "0": 0.04}
    # minimal_roi = {"0": 0.05}
    minimal_roi = {
        "20": 0.024,  # 20分钟内达到2.4% (1.2倍止损)
        "60": 0.03,  # 60分钟内3%
        "120": 0.04,  # 120分钟内4%
        "0": 0.06,  # 长期持仓6%
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # Strategy parameters
    buy_rsi = IntParameter(25, 45, default=35, space="buy")
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
                "sma90": {"color": "yellow"},
                "sma120": {"color": "blue"},
                "sma250": {"color": "red"},
                "golden_cross_marker": {
                    "type": "scatter",
                    "marker": "▲",
                    "color": "purple",
                    "markersize": 3000,
                },
                "dead_cross_marker": {
                    "type": "scatter",
                    "marker": "▼",
                    "color": "red",
                    "markersize": 3000,
                },
            },
            "subplots": {"RSI": {"rsi": {"color": "red"}}},
        }

    def has_open_trade(self, pair: str) -> LocalTrade | None:
        """
        Checks if a trade is currently open for the given pair.
        :param pair: Pair to check.
        :return: The trade object if a trade is open, otherwise None.
        """
        trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if trades:
            return trades[0]
        return None

    # 修改止损逻辑
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,  # 添加缺失参数
        **kwargs,
    ) -> float | None:  # 修正返回类型
        # 获取当前K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.03  # 默认止损

        # 找到当前时间对应的行
        current_candle = dataframe.iloc[-1]
        atr_value = current_candle["atr"]
        current_close = current_candle["close"]
        return -0.75 * atr_value / current_close  # 计算动态止损

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

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # --- RSI Divergence Calculation ---
        # 1. Price new low
        dataframe["price_new_low"] = dataframe["low"] < dataframe["low"].shift(1).rolling(14).min()

        # 2. RSI higher low
        dataframe["rsi_higher_low"] = dataframe["rsi"] > dataframe["rsi"].shift(1).rolling(14).min()

        # 3. Combine conditions to create the final divergence signal(预备状态)
        dataframe["bullish_divergence"] = dataframe["price_new_low"] & dataframe["rsi_higher_low"]

        # MACD
        # macd = ta.MACD(dataframe)
        # dataframe["macd"] = macd["macd"]
        # dataframe["macdsignal"] = macd["macdsignal"]
        # dataframe["macdhist"] = macd["macdhist"]

        # # EMA - Exponential Moving Average
        # dataframe["ema3"] = ta.EMA(dataframe, timeperiod=3)
        # dataframe["ema5"] = ta.EMA(dataframe, timeperiod=5)
        # dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        # dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        # dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        # dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)

        # SMA - Simple Moving Average
        # dataframe["sma3"] = ta.SMA(dataframe, timeperiod=3)
        # dataframe["sma5"] = ta.SMA(dataframe, timeperiod=5)
        # dataframe["sma10"] = ta.SMA(dataframe, timeperiod=10)
        # dataframe["sma21"] = ta.SMA(dataframe, timeperiod=21)
        # dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
        # dataframe["sma100"] = ta.SMA(dataframe, timeperiod=100)
        dataframe["sma90"] = ta.SMA(dataframe, timeperiod=90)
        dataframe["sma120"] = ta.SMA(dataframe, timeperiod=120)
        dataframe["sma250"] = ta.SMA(dataframe, timeperiod=250)

        # Golden/Death cross signals
        dataframe["sma90_cross_sma120_golden"] = qtpylib.crossed_above(
            dataframe["sma90"], dataframe["sma120"]
        )
        dataframe["sma90_cross_sma120_dead"] = qtpylib.crossed_below(
            dataframe["sma90"], dataframe["sma120"]
        )

        # Markers for plotting
        dataframe["golden_cross_marker"] = np.nan
        dataframe.loc[dataframe["sma90_cross_sma120_golden"], "golden_cross_marker"] = dataframe[
            "low"
        ]
        dataframe["dead_cross_marker"] = np.nan
        dataframe.loc[dataframe["sma90_cross_sma120_dead"], "dead_cross_marker"] = dataframe["high"]

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

        # 添加量能指标
        dataframe["volume_ma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_pct"] = dataframe["volume"] / dataframe["volume_ma20"]

        # 在populate_indicators中添加ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # --- 如何使用 has_open_trade ---
        pair = metadata["pair"]

        # 调用新方法
        trade = self.has_open_trade(pair)

        if trade:
            # 如果 trade 不是 None,说明有持仓
            print(f"调试信息: 交易对 {pair} 当前有一个持仓,开仓价格为 {trade.open_rate}。")
        else:
            # 如果 trade 是 None,说明没有持仓
            print(f"调试信息: 交易对 {pair} 当前没有持仓。")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        """
        开仓入场信号逻辑说明:
        本策略采用双模式开仓机制,根据ADX指标判断市场状态,分别触发趋势策略和反转策略:

        1. 趋势策略 (当ADX>25时,表明市场处于趋势行情)
           - 均线排列: sma90 > sma120 > sma250 (多头排列)
           - 价格位置: 收盘价 > sma90 (位于短期均线上方)
           - 量能确认: 当前成交量 > 20日均量线的1.2倍
           - 量能持续性: 前一根K线成交量 > 20日均量线的1.0倍
           - 趋势强度: ADX > 25

        2. 反转策略 (当ADX<20时,表明市场处于震荡行情)
           - 技术形态: 出现RSI底背离信号 (价格创新低而RSI未创新低)
           - 超卖区域: RSI < 30
           - 量能确认: 当前成交量 > 20日均量线的2.0倍
           - 量能持续性: 前一根K线成交量 > 20日均量线的1.5倍
           - 趋势强度: ADX < 20

        满足任一策略条件即触发开仓信号。
        """

        # 趋势策略条件
        trend_conditions = (
            (dataframe["sma90"] > dataframe["sma120"])
            & (qtpylib.crossed_above(dataframe["sma90"], dataframe["sma120"]))  # 添加金叉确认
            & (dataframe["close"] > dataframe["sma90"])
            & (dataframe["volume_pct"] > 1.5)  # 从1.2提高到1.5
            & (dataframe["adx"] > 20)  # 从25降至20
        )

        # 反转策略条件
        reversal_conditions = (
            dataframe["bullish_divergence"]
            & (dataframe["rsi"] < 28)  # 从30降至28
            & (dataframe["close"] < dataframe["sma250"] * 0.93)  # 添加价格位置过滤
            & (dataframe["volume_pct"] > 2.5)  # 从2.0提高到2.5
            & (dataframe["volume_pct"].shift(1) > 1.8)  # 从1.5提高到1.8
            & (dataframe["adx"] < 15)  # 从20降至15
        )

        # 合并条件
        conditions = trend_conditions | reversal_conditions
        dataframe.loc[conditions, "enter_long"] = 1

        # --- 日志记录 ---
        # 记录趋势策略开仓
        trend_signals = dataframe[trend_conditions]
        for index, row in trend_signals.iterrows():
            print(
                f"趋势策略开仓: 交易对={metadata['pair']}, "
                f"时间={row['date']}, 价格={row['close']:.2f}, ADX={row['adx']:.1f}"
            )

        # 记录反转策略开仓
        reversal_signals = dataframe[reversal_conditions]
        for index, row in reversal_signals.iterrows():
            print(
                f"反转策略开仓: 交易对={metadata['pair']}, "
                f"时间={row['date']}, 价格={row['close']:.2f}, RSI={row['rsi']:.1f}"
            )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_below(dataframe["sma90"], dataframe["sma120"]))  # Death cross
                    | (
                        qtpylib.crossed_below(dataframe["close"], dataframe["sma250"])
                    )  # Price crosses below sma250
                )
                & (dataframe["volume"] > 0)  # Ensure there is volume
            ),
            "exit_long",
        ] = 1

        return dataframe
