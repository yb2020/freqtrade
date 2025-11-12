# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from datetime import datetime

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)

import talib.abstract as ta


class MultiTimeframeMAStrategy(IStrategy):
    """
    多时间框架均线策略 - 专为数字货币优化
    
    核心逻辑：
    - 5分钟周期：判断趋势方向（MA120/MA250金叉死叉）
    - 1分钟周期：寻找精确入场点（快速均线回调）
    - 出场信号：以5分钟趋势反转为准
    - 入场信号：以1分钟回调为准
    
    优势：
    1. 大周期确认趋势，避免假突破
    2. 小周期精确入场，降低成本
    3. 大周期出场，避免被震出
    """

    INTERFACE_VERSION = 3

    # 只做多
    can_short: bool = False

    # ROI设置 - 更激进的止盈
    minimal_roi = {
        "0": 0.05,      # 5%利润立即卖出
        "30": 0.03,     # 30分钟后3%卖出
        "60": 0.02,     # 60分钟后2%卖出
        "120": 0.01,    # 120分钟后1%卖出
    }

    # 止损
    stoploss = -0.04  # -4%止损（收紧）

    # 追踪止损
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # 主时间周期：1分钟
    timeframe = "1m"

    # 只在新K线时运行
    process_only_new_candles = True

    # 使用卖出信号
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ==================== 可优化参数 ====================
    
    # 5分钟趋势判断均线（慢）
    trend_ma_fast = IntParameter(100, 150, default=120, space="buy", optimize=True)
    trend_ma_slow = IntParameter(200, 300, default=250, space="buy", optimize=True)
    
    # 1分钟入场均线（快）
    entry_ma_fast = IntParameter(5, 20, default=10, space="buy", optimize=True)
    entry_ma_slow = IntParameter(20, 50, default=30, space="buy", optimize=True)
    
    # 均线类型 (0=SMA, 1=EMA)
    ma_type = IntParameter(0, 1, default=1, space="buy", optimize=True)
    
    # 成交量过滤倍数
    volume_factor = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    
    # RSI过滤（避免超买入场）
    rsi_entry_max = IntParameter(60, 80, default=70, space="buy", optimize=True)
    rsi_exit_min = IntParameter(20, 40, default=30, space="exit", optimize=True)

    # 启动需要的K线数量
    startup_candle_count: int = 300

    # 订单类型
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # 图表配置
    plot_config = {
        "main_plot": {
            "entry_ma_fast": {"color": "blue"},
            "entry_ma_slow": {"color": "orange"},
        },
        "subplots": {
            "趋势(5m)": {
                "trend_up_5m": {"color": "green"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "成交量": {
                "volume": {"color": "gray"},
            },
        },
    }

    def informative_pairs(self):
        """
        定义需要的额外时间周期数据
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "5m") for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加技术指标
        """
        # ==================== 1分钟周期指标 ====================
        
        # 1分钟快速均线（用于入场）
        if self.ma_type.value == 0:
            dataframe['entry_ma_fast'] = ta.SMA(dataframe, timeperiod=self.entry_ma_fast.value)
            dataframe['entry_ma_slow'] = ta.SMA(dataframe, timeperiod=self.entry_ma_slow.value)
        else:
            dataframe['entry_ma_fast'] = ta.EMA(dataframe, timeperiod=self.entry_ma_fast.value)
            dataframe['entry_ma_slow'] = ta.EMA(dataframe, timeperiod=self.entry_ma_slow.value)
        
        # RSI（避免超买入场）
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # 成交量均线
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        # ATR（波动率）
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # ==================== 5分钟周期指标 ====================
        
        # 获取5分钟数据
        informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='5m')
        
        # 5分钟趋势均线
        if self.ma_type.value == 0:
            informative_5m['trend_ma_fast'] = ta.SMA(informative_5m, timeperiod=self.trend_ma_fast.value)
            informative_5m['trend_ma_slow'] = ta.SMA(informative_5m, timeperiod=self.trend_ma_slow.value)
        else:
            informative_5m['trend_ma_fast'] = ta.EMA(informative_5m, timeperiod=self.trend_ma_fast.value)
            informative_5m['trend_ma_slow'] = ta.SMA(informative_5m, timeperiod=self.trend_ma_slow.value)
        
        # 5分钟趋势方向
        informative_5m['trend_up'] = (
            informative_5m['trend_ma_fast'] > informative_5m['trend_ma_slow']
        ).astype(int)
        
        # 5分钟均线距离（判断趋势强度）
        informative_5m['trend_distance'] = (
            (informative_5m['trend_ma_fast'] - informative_5m['trend_ma_slow']) 
            / informative_5m['trend_ma_slow'] * 100
        )
        
        # 5分钟均线斜率（判断趋势加速）
        informative_5m['trend_ma_fast_slope'] = (
            informative_5m['trend_ma_fast'] - informative_5m['trend_ma_fast'].shift(3)
        ) / informative_5m['trend_ma_fast'].shift(3) * 100
        
        # 5分钟RSI
        informative_5m['rsi'] = ta.RSI(informative_5m, timeperiod=14)
        
        # 5分钟MACD（辅助判断）
        macd_5m = ta.MACD(informative_5m, fastperiod=12, slowperiod=26, signalperiod=9)
        informative_5m['macd'] = macd_5m['macd']
        informative_5m['macdsignal'] = macd_5m['macdsignal']
        informative_5m['macdhist'] = macd_5m['macdhist']
        
        # 合并5分钟数据到1分钟
        dataframe = merge_informative_pair(
            dataframe, informative_5m, self.timeframe, '5m', ffill=True
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        买入信号：放宽条件，增加交易机会
        """
        dataframe.loc[
            (
                # ========== 5分钟趋势确认（简化）==========
                # 条件1：5分钟趋势向上
                (dataframe['trend_up_5m'] == 1) &
                
                # 条件2：5分钟MACD为正（动量确认）
                (dataframe['macdhist_5m'] > 0) &
                
                # ========== 1分钟精确入场 ==========
                # 条件3：1分钟金叉（快线上穿慢线）
                (dataframe['entry_ma_fast'] > dataframe['entry_ma_slow']) &
                (dataframe['entry_ma_fast'].shift(1) <= dataframe['entry_ma_slow'].shift(1)) &
                
                # 条件4：价格在1分钟快线上方
                (dataframe['close'] > dataframe['entry_ma_fast']) &
                
                # ========== 过滤条件（简化）==========
                # 条件5：RSI不超买
                (dataframe['rsi'] < 75) &
                
                # 条件6：成交量确认
                (dataframe['volume'] > dataframe['volume_ma'] * 1.2) &
                
                # 条件7：确保数据有效
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        卖出信号：只在明确趋势反转时出场
        """
        dataframe.loc[
            (
                # 主要出场：5分钟明确死叉
                (dataframe['trend_up_5m'] == 0) &
                
                # MACD确认
                (dataframe['macdhist_5m'] < 0) &
                
                # 1分钟也确认下跌
                (dataframe['entry_ma_fast'] < dataframe['entry_ma_slow']) &
                
                # 确保成交量有效
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: 'Trade',
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        动态止损：快速止盈，严格止损
        """
        # 如果已经盈利，提升止损保护利润
        if current_profit > 0.03:  # 盈利超过3%
            return -(current_profit * 0.6)  # 保护60%利润
        elif current_profit > 0.02:  # 盈利超过2%
            return -(current_profit * 0.5)  # 保护50%利润
        elif current_profit > 0.01:  # 盈利超过1%
            return -0.005  # 移动到盈亏平衡附近
        
        # 否则使用固定止损
        return self.stoploss

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        入场前的最后确认（简化）
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        
        # 只确认5分钟趋势向上
        return last_candle['trend_up_5m'] == 1

    def confirm_trade_exit(
        self,
        pair: str,
        trade: 'Trade',
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        出场前的最后确认
        """
        # ROI和止损直接执行
        if exit_reason in ['roi', 'stop_loss', 'trailing_stop_loss']:
            return True
        
        # exit_signal也直接执行（不再二次确认）
        return True
