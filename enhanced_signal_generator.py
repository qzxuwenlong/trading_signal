"""
增强版交易信号生成模块 v2
使用所有高级技术指标 + 智能冲突检测
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from okx_data import OKXDataFetcher, get_market_data
from advanced_indicators import AdvancedIndicators, analyze_all_indicators
from conflict_detector import (
    ConflictDetector, ConflictResult, 
    SignalType, Action, detect_conflicts
)
from kronos_model import KronosPredictorWrapper, KronosPrediction, create_predictor


@dataclass
class TradingSignal:
    """交易信号 v2"""
    symbol: str
    signal_type: str  # 'long', 'short', 'neutral'
    confidence: float  # 原始置信度 0-1
    adjusted_confidence: float  # 调整后置信度
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    signal_analysis: Optional[ConflictResult] = None
    indicators: Dict = field(default_factory=dict)
    kronos_prediction: Optional[KronosPrediction] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_reward_ratio: float = 0.0
    suggested_action: str = ""  # 建议动作


class EnhancedSignalGenerator:
    """增强版交易信号生成器"""
    
    def __init__(
        self,
        kronos_predictor: Optional[KronosPredictorWrapper] = None,
        confidence_threshold: float = 0.5
    ):
        """
        初始化
        
        Args:
            kronos_predictor: Kronos 预测器
            confidence_threshold: 置信度阈值
        """
        self.kronos_predictor = kronos_predictor
        self.confidence_threshold = confidence_threshold
        self.fetcher = OKXDataFetcher()
    
    def generate_signal(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "15m",
        lookback: int = 400,
        pred_len: int = 6
    ) -> TradingSignal:
        """
        生成交易信号
        """
        # 1. 获取市场数据
        print(f"Fetching {symbol} data...")
        df = get_market_data(symbol, timeframe, lookback + 100)
        
        if df.empty or len(df) < 100:
            return TradingSignal(
                symbol=symbol,
                signal_type='neutral',
                confidence=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                reasons=["Insufficient data"]
            )
        
        # 2. 计算所有技术指标
        print("Analyzing all technical indicators...")
        all_indicators = analyze_all_indicators(df)
        
        # 3. Kronos 预测
        kronos_pred = None
        if self.kronos_predictor:
            print("Running Kronos prediction...")
            kronos_pred = self.kronos_predictor.predict(
                df.tail(lookback),
                pred_len=pred_len
            )
        
        # 4. 生成信号
        signal = self._analyze_and_generate(
            symbol=symbol,
            indicators=all_indicators,
            kronos_pred=kronos_pred,
            df=df
        )
        
        return signal
    
    def _analyze_and_generate(
        self,
        symbol: str,
        indicators: Dict,
        kronos_pred: Optional[KronosPrediction],
        df: pd.DataFrame
    ) -> TradingSignal:
        """
        分析并生成信号 v2
        使用新的冲突检测逻辑
        """
        reasons = []
        warnings = []
        long_score = 0
        short_score = 0
        
        current_price = indicators['current_price']
        
        # ==================== 智能冲突检测 ====================
        conflict_result = detect_conflicts(indicators)
        
        # 将冲突检测的结果整合到信号生成中
        # 冲突检测已经分析了趋势、动量、量价
        # 我们需要将其与单独的指标分析结合
        
        # ==================== 基于冲突检测的初步信号 ====================
        
        # 从冲突检测获取建议动作
        if conflict_result.final_action == Action.GO_LONG:
            long_score += 0.2
            reasons.append("冲突检测建议做多")
        elif conflict_result.final_action == Action.GO_SHORT:
            short_score += 0.2
            reasons.append("冲突检测建议做空")
        elif conflict_result.final_action == Action.WAIT:
            warnings.append("冲突检测建议等待")
        elif conflict_result.final_action == Action.REDUCE_POSITION:
            warnings.append("冲突检测建议减仓")
        
        # 收集冲突检测的理由和警告
        reasons.extend(conflict_result.all_reasons)
        warnings.extend(conflict_result.all_warnings)
        
        # ==================== 其他指标分析 ====================
        
        # 1. 布林带分析
        bollinger = indicators['bollinger']
        percent_b = bollinger['percent_b']
        
        if percent_b < 0.2:
            long_score += 0.08
            reasons.append(f"Bollinger %B={percent_b:.2f} (near lower band)")
        elif percent_b > 0.8:
            short_score += 0.08
            reasons.append(f"Bollinger %B={percent_b:.2f} (near upper band)")
        
        # 2. Keltner Channel 分析
        keltner = indicators['keltner']
        if current_price < keltner['lower']:
            long_score += 0.06
            reasons.append("Price below Keltner lower band")
        elif current_price > keltner['upper']:
            short_score += 0.06
            reasons.append("Price above Keltner upper band")
        
        # 3. Donchian Channel 分析
        donchian = indicators['donchian']
        if current_price >= donchian['upper'] * 0.99:
            long_score += 0.05
            reasons.append("Price near Donchian upper band (breakout)")
        elif current_price <= donchian['lower'] * 1.01:
            short_score += 0.05
            reasons.append("Price near Donchian lower band (breakdown)")
        
        # 4. 支撑阻力分析
        sr_levels = indicators['support_resistance']
        nearest_support = None
        nearest_resistance = None
        
        for level in sr_levels:
            if level['type'] == 'support' and level['price'] < current_price:
                if nearest_support is None or level['price'] > nearest_support['price']:
                    nearest_support = level
            elif level['type'] == 'resistance' and level['price'] > current_price:
                if nearest_resistance is None or level['price'] < nearest_resistance['price']:
                    nearest_resistance = level
        
        if nearest_support:
            support_distance = (current_price - nearest_support['price']) / current_price
            if support_distance < 0.01:
                long_score += 0.1
                reasons.append(f"Near support ${nearest_support['price']:.2f}")
        
        if nearest_resistance:
            resistance_distance = (nearest_resistance['price'] - current_price) / current_price
            if resistance_distance < 0.01:
                short_score += 0.1
                reasons.append(f"Near resistance ${nearest_resistance['price']:.2f}")
        
        # 5. K线形态分析
        patterns = indicators['candlestick_patterns']
        for pattern in patterns:
            if pattern['type'] == 'bullish':
                long_score += 0.08 * pattern['strength']
                reasons.append(f"Bullish pattern: {pattern['name']}")
            elif pattern['type'] == 'bearish':
                short_score += 0.08 * pattern['strength']
                reasons.append(f"Bearish pattern: {pattern['name']}")
        
        # 6. 斐波那契回撤分析
        fibonacci = indicators['fibonacci']
        for ratio, price in fibonacci.items():
            if ratio in [0.382, 0.5, 0.618]:
                distance = abs(current_price - price) / current_price
                if distance < 0.005:
                    if current_price > price:
                        long_score += 0.05
                        reasons.append(f"Price above Fibonacci {ratio:.1%} (${price:.2f})")
                    else:
                        short_score += 0.05
                        reasons.append(f"Price below Fibonacci {ratio:.1%} (${price:.2f})")
        
        # 7. 枢轴点分析
        pivot = indicators['pivot']
        if current_price > pivot['r1']:
            long_score += 0.03
        elif current_price < pivot['s1']:
            short_score += 0.03
        
        # 8. 趋势线分析
        trendline = indicators['trendline']
        if trendline['type'] == 'up' and trendline['strength'] > 0.5:
            long_score += 0.06
            reasons.append(f"Uptrend (strength: {trendline['strength']:.2f})")
        elif trendline['type'] == 'down' and trendline['strength'] > 0.5:
            short_score += 0.06
            reasons.append(f"Downtrend (strength: {trendline['strength']:.2f})")
        
        # 9. Kronos 预测分析
        if kronos_pred:
            if kronos_pred.trend == 'up':
                long_score += 0.15
                reasons.append(f"Kronos predicts uptrend ({kronos_pred.change_pct:+.2f}%)")
            elif kronos_pred.trend == 'down':
                short_score += 0.15
                reasons.append(f"Kronos predicts downtrend ({kronos_pred.change_pct:+.2f}%)")
        
        # ==================== 综合判断 ====================
        
        confidence = 0
        signal_type = 'neutral'
        entry_price = current_price
        stop_loss = current_price * 0.97
        take_profit = current_price * 1.03
        
        # 初步判断信号方向
        if long_score > short_score:
            signal_type = 'long'
            confidence = long_score
        elif short_score > long_score:
            signal_type = 'short'
            confidence = short_score
        else:
            signal_type = 'neutral'
            confidence = 0
        
        # ==================== 应用冲突检测调整 ====================
        
        # 根据冲突检测结果调整置信度
        adjusted_confidence = confidence * (1 + conflict_result.final_adjustment)
        adjusted_confidence = max(0, min(1, adjusted_confidence))
        
        # 如果冲突检测建议等待，且没有强烈信号，设为中性
        if conflict_result.final_action == Action.WAIT and confidence < 0.3:
            signal_type = 'neutral'
            reasons.append("Conflict detection:建议等待确认")
        
        # 如果有顶背离警告，降低做多信号
        if any('顶背离' in w for w in warnings) and signal_type == 'long':
            adjusted_confidence *= 0.8
            reasons.append("顶背离警告，降低做多置信度")
        
        # 如果有底背离信号，增加做多信号
        if any('底背离' in r for r in reasons) and signal_type != 'short':
            signal_type = 'long'
            adjusted_confidence = max(adjusted_confidence, 0.6)
        
        # ==================== 计算止损止盈 ====================
        
        atr = indicators['atr']['value']
        
        if signal_type == 'long':
            if nearest_support:
                stop_loss = nearest_support['price'] * 0.99
            else:
                stop_loss = current_price - atr * 2
            
            if nearest_resistance:
                take_profit = nearest_resistance['price']
            else:
                take_profit = current_price + atr * 3
                
        elif signal_type == 'short':
            if nearest_resistance:
                stop_loss = nearest_resistance['price'] * 1.01
            else:
                stop_loss = current_price + atr * 2
            
            if nearest_support:
                take_profit = nearest_support['price']
            else:
                take_profit = current_price - atr * 3
        
        # 计算风险回报比
        if signal_type == 'long':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        elif signal_type == 'short':
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        else:
            risk = 1
            reward = 1
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # 确定建议动作
        suggested_action = self._get_suggested_action(
            signal_type, adjusted_confidence, conflict_result, risk_reward_ratio
        )
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            warnings=warnings,
            signal_analysis=conflict_result,
            indicators=indicators,
            kronos_prediction=kronos_pred,
            risk_reward_ratio=risk_reward_ratio,
            suggested_action=suggested_action
        )
    
    def _get_suggested_action(
        self,
        signal_type: str,
        confidence: float,
        conflict_result: ConflictResult,
        risk_reward: float
    ) -> str:
        """
        根据信号和冲突情况，给出最终建议
        """
        # 如果冲突检测有明确建议
        if conflict_result.final_action == Action.WAIT:
            return "等待确认后入场"
        
        if conflict_result.final_action == Action.REDUCE_POSITION:
            return "减仓观望"
        
        # 如果信号弱或风险回报不好
        if confidence < 0.4:
            return "信号较弱，建议观望"
        
        if risk_reward < 1:
            return "风险回报比不佳，建议观望"
        
        # 正常信号
        if signal_type == 'long':
            if confidence > 0.7:
                return "强烈做多信号"
            else:
                return "做多信号"
        elif signal_type == 'short':
            if confidence > 0.7:
                return "强烈做空信号"
            else:
                return "做空信号"
        
        return "观望"
    
    def generate_signals_for_multiple(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        lookback: int = 400,
        pred_len: int = 6
    ) -> List[TradingSignal]:
        """
        为多个交易对生成信号
        """
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol, timeframe, lookback, pred_len)
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal for {symbol}: {e}")
        
        return signals


def format_signal(signal: TradingSignal) -> str:
    """
    格式化信号输出 v2
    """
    emoji_map = {'long': '🟢', 'short': '🔴', 'neutral': '⚪'}
    emoji = emoji_map.get(signal.signal_type, '⚪')
    
    output = f"""
{emoji} {signal.symbol} Signal: {signal.signal_type.upper()}
{'='*60}
Entry Price:        ${signal.entry_price:.2f}
Stop Loss:          ${signal.stop_loss:.2f}
Take Profit:        ${signal.take_profit:.2f}
Raw Confidence:     {signal.confidence:.2%}
Adjusted Confidence: {signal.adjusted_confidence:.2%}
Risk/Reward:        {signal.risk_reward_ratio:.2f}
{'='*60}
"""
    
    # 显示冲突分析结果
    if signal.signal_analysis:
        output += "📊 SIGNAL ANALYSIS:\n"
        output += "-"*60 + "\n"
        
        if signal.signal_analysis.trend_analysis:
            ta = signal.signal_analysis.trend_analysis
            output += f"趋势分析: {ta.signal_type.value}\n"
            for r in ta.reasons:
                output += f"  • {r}\n"
        
        if signal.signal_analysis.momentum_analysis:
            ma = signal.signal_analysis.momentum_analysis
            output += f"动量分析: {ma.signal_type.value}\n"
            for r in ma.reasons:
                output += f"  • {r}\n"
        
        if signal.signal_analysis.volume_analysis:
            va = signal.signal_analysis.volume_analysis
            output += f"量价分析: {va.signal_type.value}\n"
            for r in va.reasons:
                output += f"  • {r}\n"
        
        output += "-"*60 + "\n"
    
    # 显示警告
    if signal.warnings:
        output += "⚠️  WARNINGS:\n"
        for w in signal.warnings:
            output += f"  ⚠️ {w}\n"
        output += "-"*60 + "\n"
    
    # 显示理由
    output += "Reasons:\n"
    for reason in signal.reasons:
        output += f"  • {reason}\n"
    
    # 显示 Kronos 预测
    if signal.kronos_prediction:
        output += f"\nKronos Prediction:"
        output += f"\n  Trend: {signal.kronos_prediction.trend}"
        output += f"\n  Change: {signal.kronos_prediction.change_pct:+.2f}%"
    
    # 显示建议
    output += f"\n{'='*60}"
    output += f"\n💡 Suggestion: {signal.suggested_action}"
    output += f"\n{'='*60}"
    
    return output


if __name__ == "__main__":
    # 测试
    print("Testing Enhanced Signal Generator...")
    
    generator = EnhancedSignalGenerator(kronos_predictor=None)
    
    # 单个交易对
    signal = generator.generate_signal("BTC-USDT", "15m", 400, 6)
    print(format_signal(signal))
