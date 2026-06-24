"""
交易信号生成模块
结合技术指标和 Kronos 预测生成交易信号
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
from indicators import TechnicalIndicators, analyze_technicals
from kronos_model import KronosPredictorWrapper, KronosPrediction, create_predictor


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: str  # 'long', 'short', 'neutral'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str] = field(default_factory=list)
    indicators: Dict = field(default_factory=dict)
    kronos_prediction: Optional[KronosPrediction] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_reward_ratio: float = 0.0


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(
        self,
        kronos_predictor: Optional[KronosPredictorWrapper] = None,
        confidence_threshold: float = 0.6
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
        
        Args:
            symbol: 交易对
            timeframe: K线周期
            lookback: 回看长度
            pred_len: 预测长度
            
        Returns:
            交易信号
        """
        # 1. 获取市场数据
        print(f"Fetching {symbol} data...")
        df = get_market_data(symbol, timeframe, lookback + 100)
        
        if df.empty or len(df) < lookback:
            return TradingSignal(
                symbol=symbol,
                signal_type='neutral',
                confidence=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                reasons=["Insufficient data"]
            )
        
        # 2. 计算技术指标
        print("Analyzing technical indicators...")
        tech_analyzer = TechnicalIndicators(df)
        key_levels = tech_analyzer.get_key_levels()
        
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
            current_price=key_levels['current_price'],
            key_levels=key_levels,
            kronos_pred=kronos_pred,
            df=df
        )
        
        return signal
    
    def _analyze_and_generate(
        self,
        symbol: str,
        current_price: float,
        key_levels: Dict,
        kronos_pred: Optional[KronosPrediction],
        df: pd.DataFrame
    ) -> TradingSignal:
        """
        分析并生成信号
        """
        reasons = []
        long_score = 0
        short_score = 0
        
        # === 技术指标分析 ===
        
        # 1. 支撑阻力分析
        if key_levels['nearest_support']:
            support = key_levels['nearest_support']
            support_distance = (current_price - support['price']) / current_price
            
            if support_distance < 0.01:  # 接近支撑位
                long_score += 0.3
                reasons.append(f"Near support ${support['price']:.2f} (strength: {support['strength']:.2f})")
        
        if key_levels['nearest_resistance']:
            resistance = key_levels['nearest_resistance']
            resistance_distance = (resistance['price'] - current_price) / current_price
            
            if resistance_distance < 0.01:  # 接近阻力位
                short_score += 0.3
                reasons.append(f"Near resistance ${resistance['price']:.2f} (strength: {resistance['strength']:.2f})")
        
        # 2. 布林带分析
        bollinger = key_levels['bollinger_position']
        percent_b = bollinger['percent_b']
        
        if percent_b < 0.2:  # 接近下轨
            long_score += 0.2
            reasons.append(f"Bollinger %B={percent_b:.2f} (near lower band)")
        elif percent_b > 0.8:  # 接近上轨
            short_score += 0.2
            reasons.append(f"Bollinger %B={percent_b:.2f} (near upper band)")
        
        # 3. RSI 分析
        rsi = key_levels['rsi']
        
        if rsi < 30:  # 超卖
            long_score += 0.2
            reasons.append(f"RSI={rsi:.2f} (oversold)")
        elif rsi > 70:  # 超买
            short_score += 0.2
            reasons.append(f"RSI={rsi:.2f} (overbought)")
        
        # 4. 趋势线分析
        trend = key_levels['trend']
        if trend['type'] == 'up' and trend['strength'] > 0.5:
            long_score += 0.1
            reasons.append(f"Uptrend (strength: {trend['strength']:.2f})")
        elif trend['type'] == 'down' and trend['strength'] > 0.5:
            short_score += 0.1
            reasons.append(f"Downtrend (strength: {trend['strength']:.2f})")
        
        # 5. 谐波形态分析
        if key_levels['harmonic']:
            for h in key_levels['harmonic']:
                if h['direction'] == 'bullish' and h['completion'] > 0.7:
                    long_score += 0.15
                    reasons.append(f"Bullish {h['type']} (completion: {h['completion']:.0%})")
                elif h['direction'] == 'bearish' and h['completion'] > 0.7:
                    short_score += 0.15
                    reasons.append(f"Bearish {h['type']} (completion: {h['completion']:.0%})")
        
        # === Kronos 预测分析 ===
        if kronos_pred:
            if kronos_pred.trend == 'up':
                long_score += 0.25
                reasons.append(f"Kronos predicts uptrend ({kronos_pred.change_pct:+.2f}%)")
            elif kronos_pred.trend == 'down':
                short_score += 0.25
                reasons.append(f"Kronos predicts downtrend ({kronos_pred.change_pct:+.2f}%)")
        
        # === 综合判断 ===
        confidence = 0
        signal_type = 'neutral'
        entry_price = current_price
        
        if long_score > short_score and long_score >= self.confidence_threshold:
            signal_type = 'long'
            confidence = min(long_score, 1.0)
            
            # 计算止损止盈
            if key_levels['nearest_support']:
                stop_loss = key_levels['nearest_support']['price'] * 0.99
            else:
                stop_loss = current_price * 0.97
            
            if key_levels['nearest_resistance']:
                take_profit = key_levels['nearest_resistance']['price']
            else:
                take_profit = current_price * 1.03
                
        elif short_score > long_score and short_score >= self.confidence_threshold:
            signal_type = 'short'
            confidence = min(short_score, 1.0)
            
            # 计算止损止盈
            if key_levels['nearest_resistance']:
                stop_loss = key_levels['nearest_resistance']['price'] * 1.01
            else:
                stop_loss = current_price * 1.03
            
            if key_levels['nearest_support']:
                take_profit = key_levels['nearest_support']['price']
            else:
                take_profit = current_price * 0.97
        else:
            signal_type = 'neutral'
            confidence = max(long_score, short_score)
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.03
            reasons.append("No clear signal")
        
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
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            indicators=key_levels,
            kronos_prediction=kronos_pred,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def generate_signals_for_multiple(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        lookback: int = 400,
        pred_len: int = 6
    ) -> List[TradingSignal]:
        """
        为多个交易对生成信号
        
        Args:
            symbols: 交易对列表
            timeframe: K线周期
            lookback: 回看长度
            pred_len: 预测长度
            
        Returns:
            交易信号列表
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
    格式化信号输出
    """
    emoji_map = {'long': '🟢', 'short': '🔴', 'neutral': '⚪'}
    emoji = emoji_map.get(signal.signal_type, '⚪')
    
    output = f"""
{emoji} {signal.symbol} Signal: {signal.signal_type.upper()}
{'='*50}
Entry Price:  ${signal.entry_price:.2f}
Stop Loss:    ${signal.stop_loss:.2f}
Take Profit:  ${signal.take_profit:.2f}
Confidence:   {signal.confidence:.2%}
Risk/Reward:  {signal.risk_reward_ratio:.2f}
{'='*50}
Reasons:
"""
    for reason in signal.reasons:
        output += f"  • {reason}\n"
    
    if signal.kronos_prediction:
        output += f"\nKronos Prediction:"
        output += f"\n  Trend: {signal.kronos_prediction.trend}"
        output += f"\n  Change: {signal.kronos_prediction.change_pct:+.2f}%"
    
    return output


if __name__ == "__main__":
    # 测试
    print("Testing Signal Generator...")
    
    generator = SignalGenerator()
    
    # 单个交易对
    signal = generator.generate_signal("BTC-USDT", "15m", 400, 6)
    print(format_signal(signal))
