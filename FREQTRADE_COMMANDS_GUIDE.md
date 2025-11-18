# Freqtrade 核心操作指南

本文档提取了所有 Freqtrade 的核心操作命令，用于快速参考。

## 1. 访问与连接

### Web UI
- **URL**: http://127.0.0.1:8080
- **用户名**: `freqtrader`
- **密码**: `freqtrader`

### API 端点
```bash
# 检查机器人状态
curl -u freqtrader:freqtrader http://127.0.0.1:8080/api/v1/status

# 查看虚拟钱包余额
curl -u freqtrader:freqtrader http://127.0.0.1:8080/api/v1/balance

# 查看当前配置
curl -u freqtrader:freqtrader http://127.0.0.1:8080/api/v1/show_config
```

## 2. 核心交易命令

### 启动交易机器人 (模拟盘)
```bash
conda run -n freqtrade freqtrade trade -c config_sample.json
```

### 停止机器人
```bash
# 方法1: 在运行的终端中使用 Ctrl+C

# 方法2: 使用 pkill 强制停止
pkill -f "freqtrade trade"
```

## 3. 数据与策略

### 下载历史数据
```bash
# 示例：下载币安 BTC/USDT 和 ETH/USDT 的5分钟K线，最近3天的数据
conda run -n freqtrade freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT \
    --timeframe 5m \
    --days 3 \
    --config config_sample.json
```

### 列出可用策略
```bash
# 查看 strategies/ice 目录下的所有策略
conda run -n freqtrade freqtrade list-strategies --strategy-path strategies/ice
```

## 4. 回测流程

### 运行回测
```bash
# 示例：回测 SimpleMAStrategy 策略在指定时间范围内的数据
conda run -n freqtrade freqtrade backtesting \
    --config config_sample.json \
    --strategy SimpleMAStrategy \
    --timerange 20251113-20251114 \
    --export trades
```

### 查看回测结果 (图表UI)
此命令会启动一个本地服务，让你在浏览器中交互式地查看回测图表。
```bash
conda run -n freqtrade freqtrade webserver --config config_sample.json
```

### 查看回测结果 (文本)
```bash
conda run -n freqtrade freqtrade backtesting-show
```

## 5. 参数优化 (Hyperopt)

### 运行超参数优化
```bash
# 示例：为 SimpleMAStrategy 策略的买入(buy)和卖出(sell)参数空间运行100轮优化
conda run -n freqtrade freqtrade hyperopt \
    --config config_sample.json \
    --strategy SimpleMAStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --epochs 100 \
    --spaces buy sell
```

## 6. 配置与路径

### 切换到真实交易
⚠️ **警告：请确保策略经过充分测试后再使用真实资金！**

1.  在 `config_sample.json` 文件中，将 `"dry_run"` 的值从 `true` 修改为 `false`。
2.  在配置文件中填入你真实的交易所 API Key 和 Secret。

### 重要文件路径
- **配置文件**: `config_sample.json`
- **策略目录**: `strategies/ice/`
- **数据目录**: `user_data/data/binance/`
- **回测结果**: `user_data/backtest_results/`
- **日志目录**: `user_data/logs/`
