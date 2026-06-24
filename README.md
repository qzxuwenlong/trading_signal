# Kronos Trading Signal Generator

基于 Kronos AI 模型和传统技术分析的交易信号生成系统。

## 功能特性

- **多资产支持**: 加密货币 (BTC, ETH, SOL)、美股指数、大宗商品 (黄金、原油)
- **技术指标分析**: 支撑阻力位、布林带、RSI、趋势线、谐波形态
- **Kronos AI 预测**: 使用 Kronos 基础模型预测未来 K 线走势
- **智能信号生成**: 结合技术分析和 AI 预测生成高置信度交易信号
- **Web UI**: 可视化界面，实时查看信号和指标

## 安装

```bash
cd trading_signal
pip install -r requirements.txt
```

## 使用方法

### 命令行模式

```bash
# 获取 BTC 交易信号
python main.py --symbol BTC-USDT --timeframe 15m

# 不使用 Kronos 模型（仅技术分析）
python main.py --symbol ETH-USDT --no-kronos

# 自定义参数
python main.py --symbol XAU-USDT --timeframe 1H --lookback 200 --pred-len 12
```

### Web UI 模式

```bash
# 启动 Web 服务器
python main.py --web

# 不使用 Kronos 模型
python main.py --web --no-kronos
```

然后访问 http://localhost:8888

## 支持的交易对

| 交易对 | 资产类型 | 说明 |
|--------|----------|------|
| BTC-USDT | Crypto | 比特币 |
| ETH-USDT | Crypto | 以太坊 |
| SOL-USDT | Crypto | Solana |
| SP500-USDT | Stock | 标普500 |
| NASDAQ-USDT | Stock | 纳斯达克 |
| XAU-USDT | Commodity | 黄金 |
| XAG-USDT | Commodity | 白银 |
| WTI-USDT | Commodity | WTI原油 |

## 信号生成逻辑

### 技术分析 (60% 权重)

1. **支撑阻力位**: 接近支撑位做多，接近阻力位做空
2. **布林带**: %B < 0.2 做多，%B > 0.8 做空
3. **RSI**: RSI < 30 超卖做多，RSI > 70 超买做空
4. **趋势线**: 强上升趋势做多，强下降趋势做空
5. **谐波形态**: 看涨形态做多，看跌形态做空

### Kronos AI 预测 (40% 权重)

- 预测未来 N 根 K 线走势
- 上升趋势 → 做多信号
- 下降趋势 → 做空信号

## API 接口

### 获取交易信号

```http
POST /api/signal
Content-Type: application/json

{
    "symbol": "BTC-USDT",
    "timeframe": "15m",
    "lookback": 400,
    "pred_len": 6
}
```

### 批量获取信号

```http
POST /api/signals/batch
Content-Type: application/json

{
    "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    "timeframe": "15m"
}
```

### 获取技术指标

```http
POST /api/indicators
Content-Type: application/json

{
    "symbol": "BTC-USDT",
    "timeframe": "15m"
}
```

## 配置

编辑 `config.py` 修改:

- OKX API 地址
- 交易对配置
- 技术指标参数
- Kronos 模型配置
- Web UI 端口

## 注意事项

1. 本系统仅供学习和研究使用，不构成投资建议
2. 交易有风险，请谨慎操作
3. Kronos 模型需要 GPU 加速才能获得更好的性能
4. 建议在模拟盘上测试后再用于实盘

## License

MIT License
