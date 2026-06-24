"""检测指标冲突"""
import sys
sys.path.insert(0, ".")
from okx_data import get_market_data
from advanced_indicators import analyze_all_indicators

df = get_market_data('BTC-USDT', '15m', 200)
ind = analyze_all_indicators(df)

print("=== 指标信号分析 ===\n")

# 趋势类
print("【趋势类】")
ema_signal = "做多" if ind["current_price"] > ind["ema"]["value"] else "做空"
print(f"  EMA: {ema_signal}")
macd_signal = "做多" if ind["macd"]["histogram"] > 0 else "做空"
print(f"  MACD: {macd_signal}")
adx_signal = "做多" if ind["adx"]["plus_di"] > ind["adx"]["minus_di"] else "做空"
print(f"  ADX: {adx_signal} (ADX={ind['adx']['adx']:.1f})")
sar_signal = "做多" if ind["parabolic_sar"]["direction"] == "long" else "做空"
print(f"  SAR: {sar_signal}")

# 动量类
print("\n【动量类】")
rsi = ind["rsi"]
rsi_signal = "超卖做多" if rsi < 30 else ("超买做空" if rsi > 70 else "中性")
print(f"  RSI: {rsi_signal} ({rsi:.1f})")
stoch = ind["stochastic"]
stoch_signal = "超卖做多" if stoch["k"] < 20 else ("超买做空" if stoch["k"] > 80 else "中性")
print(f"  KDJ: {stoch_signal} (K={stoch['k']:.1f})")
cci = ind["cci"]["value"]
cci_signal = "超卖做多" if cci < -100 else ("超买做空" if cci > 100 else "中性")
print(f"  CCI: {cci_signal} ({cci:.1f})")
wr = ind["williams_r"]["value"]
wr_signal = "超卖做多" if wr < -80 else ("超买做空" if wr > -20 else "中性")
print(f"  Williams: {wr_signal} ({wr:.1f})")
mfi = ind["mfi"]["value"]
mfi_signal = "超卖做多" if mfi < 20 else ("超买做空" if mfi > 80 else "中性")
print(f"  MFI: {mfi_signal} ({mfi:.1f})")

# 波动类
print("\n【波动类】")
bb = ind["bollinger"]
bb_signal = "做多" if bb["percent_b"] < 0.2 else ("做空" if bb["percent_b"] > 0.8 else "中性")
print(f"  布林带: {bb_signal} (%B={bb['percent_b']:.2f})")

# K线形态
print("\n【K线形态】")
patterns = ind["candlestick_patterns"]
if patterns:
    for p in patterns:
        print(f"  {p['name']}: {'看涨' if p['type'] == 'bullish' else '看跌'}")
else:
    print("  无形态")

# 冲突检测
print("\n" + "="*50)
print("【冲突检测】")

# 趋势类冲突
trend_long = 0
trend_short = 0
if ema_signal == "做多": trend_long += 1
else: trend_short += 1
if macd_signal == "做多": trend_long += 1
else: trend_short += 1
if adx_signal == "做多": trend_long += 1
else: trend_short += 1
if sar_signal == "做多": trend_long += 1
else: trend_short += 1

print(f"\n趋势类: {trend_long}做多 vs {trend_short}做空")
if trend_long > 0 and trend_short > 0:
    print(f"  ⚠️ 冲突! 部分指标看多，部分看空")
else:
    print(f"  ✅ 一致")

# 动量类冲突
mom_long = 0
mom_short = 0
mom_neutral = 0
for s in [rsi_signal, stoch_signal, cci_signal, wr_signal, mfi_signal]:
    if "做多" in s: mom_long += 1
    elif "做空" in s: mom_short += 1
    else: mom_neutral += 1

print(f"\n动量类: {mom_long}做多 vs {mom_short}做空 vs {mom_neutral}中性")
if mom_long > 0 and mom_short > 0:
    print(f"  ⚠️ 冲突!")
elif mom_long > 0:
    print(f"  ✅ 看多一致")
elif mom_short > 0:
    print(f"  ✅ 看空一致")
else:
    print(f"  ⚪ 全部中性")

# 总结
print("\n" + "="*50)
total_long = trend_long + mom_long
total_short = trend_short + mom_short
print(f"总计: {total_long}做多 vs {total_short}做空")
if total_long > total_short:
    print("结论: 偏多")
elif total_short > total_long:
    print("结论: 偏空")
else:
    print("结论: 观望")
