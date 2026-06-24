"""
交易信号系统配置
"""
import os

# OKX API 配置
OKX_BASE_URL = "https://www.okx.com"

# 代理配置
PROXY_CONFIG = {
    "enabled": True,
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
    "verify_ssl": False,
}

# 支持的交易对
TRADING_PAIRS = {
    # 加密货币
    "BTC-USDT": {"name": "比特币", "category": "crypto"},
    "ETH-USDT": {"name": "以太坊", "category": "crypto"},
    "SOL-USDT": {"name": "Solana", "category": "crypto"},
    
    # 美股 (通过USDT合约)
    "SP500-USDT": {"name": "标普500", "category": "stock"},
    "NASDAQ-USDT": {"name": "纳斯达克", "category": "stock"},
    "DOW-USDT": {"name": "道琼斯", "category": "stock"},
    
    # 大宗商品
    "XAU-USDT": {"name": "黄金", "category": "commodity"},
    "XAG-USDT": {"name": "白银", "category": "commodity"},
    "WTI-USDT": {"name": "原油(WTI)", "category": "commodity"},
}

# 默认交易对
DEFAULT_SYMBOL = "BTC-USDT"

# K线周期
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
}

DEFAULT_TIMEFRAME = "15m"

# Kronos 模型配置
KRONOS_CONFIG = {
    "model": "NeoQuasar/Kronos-base",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "max_context": 512,
    "device": "cpu",  # 或 "cuda"
}

# 技术指标参数
INDICATOR_PARAMS = {
    "bollinger": {
        "period": 20,
        "std_dev": 2,
    },
    "rsi": {
        "period": 14,
    },
    "support_resistance": {
        "lookback": 100,
        "threshold": 0.02,  # 2%
    },
    "trendline": {
        "lookback": 50,
    },
    "harmonic": {
        "min_swing": 0.05,  # 最小摆动幅度 5%
    },
}

# 信号生成参数
SIGNAL_PARAMS = {
    "lookback": 400,  # Kronos 回看长度
    "pred_len": 6,    # 预测长度
    "confidence_threshold": 0.6,  # 置信度阈值
}

# Web UI 配置
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8888,
    "debug": True,
}
