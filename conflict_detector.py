"""
冲突检测模块 v2
区分：真正的冲突 vs 反转信号 vs 背离预警
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    TREND_CONTINUATION = "trend_continuation"  # 趋势延续
    REVERSAL = "reversal"                      # 反转信号
    DIVERGENCE = "divergence"                  # 背离
    CONFUSION = "confusion"                    # 真正的冲突（方向不明）


class Action(Enum):
    """建议动作"""
    GO_LONG = "go_long"           # 做多
    GO_SHORT = "go_short"         # 做空
    WAIT = "wait"                 # 等待确认
    REDUCE_POSITION = "reduce"    # 减仓
    NO_ACTION = "no_action"       # 不操作


@dataclass
class SignalAnalysis:
    """信号分析结果"""
    signal_type: SignalType
    action: Action
    confidence_adjustment: float  # 置信度调整 (-0.3 ~ +0.2)
    reasons: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConflictResult:
    """冲突检测结果"""
    # 各类信号分析
    trend_analysis: Optional[SignalAnalysis] = None
    momentum_analysis: Optional[SignalAnalysis] = None
    volume_analysis: Optional[SignalAnalysis] = None
    
    # 综合结果
    final_adjustment: float = 1.0
    final_action: Action = Action.NO_ACTION
    all_reasons: List[str] = field(default_factory=list)
    all_warnings: List[str] = field(default_factory=list)


class ConflictDetector:
    """冲突检测器 v2"""
    
    def __init__(self):
        pass
    
    def analyze(self, indicators: Dict) -> ConflictResult:
        """
        分析所有指标，返回综合结果
        """
        result = ConflictResult()
        
        # 1. 趋势类分析
        result.trend_analysis = self._analyze_trend(indicators)
        
        # 2. 动量类分析
        result.momentum_analysis = self._analyze_momentum(indicators)
        
        # 3. 量价分析
        result.volume_analysis = self._analyze_volume(indicators)
        
        # 4. 综合判断
        self._combine_analyses(result)
        
        return result
    
    def _analyze_trend(self, indicators: Dict) -> SignalAnalysis:
        """
        趋势类分析
        关键：区分趋势延续 vs 趋势反转信号
        """
        reasons = []
        warnings = []
        
        # 收集趋势指标信号
        signals = {}
        
        # EMA
        ema_bullish = indicators['current_price'] > indicators['ema']['value']
        signals['EMA'] = 'bullish' if ema_bullish else 'bearish'
        
        # MACD
        macd_bullish = indicators['macd']['histogram'] > 0
        signals['MACD'] = 'bullish' if macd_bullish else 'bearish'
        
        # MACD 金叉/死叉
        macd_cross_bullish = indicators['macd']['histogram'] > 0 and indicators['macd']['prev_histogram'] <= 0
        macd_cross_bearish = indicators['macd']['histogram'] < 0 and indicators['macd']['prev_histogram'] >= 0
        
        # ADX
        adx = indicators['adx']
        if adx['adx'] > 25:
            if adx['plus_di'] > adx['minus_di']:
                signals['ADX'] = 'bullish'
            else:
                signals['ADX'] = 'bearish'
        
        # SAR
        signals['SAR'] = 'bullish' if indicators['parabolic_sar']['direction'] == 'long' else 'bearish'
        
        # Ichimoku
        price = indicators['current_price']
        ichimoku = indicators['ichimoku']
        if price > ichimoku['senkou_a'] and price > ichimoku['senkou_b']:
            signals['Ichimoku'] = 'bullish'
        elif price < ichimoku['senkou_a'] and price < ichimoku['senkou_b']:
            signals['Ichimoku'] = 'bearish'
        
        # 统计
        bullish_count = sum(1 for v in signals.values() if v == 'bullish')
        bearish_count = sum(1 for v in signals.values() if v == 'bearish')
        total = len(signals)
        
        # 判断趋势方向
        if bullish_count > bearish_count and bullish_count >= total * 0.6:
            trend_direction = 'bullish'
            reasons.append(f"趋势看多: {bullish_count}/{total} 指标看多")
        elif bearish_count > bullish_count and bearish_count >= total * 0.6:
            trend_direction = 'bearish'
            reasons.append(f"趋势看空: {bearish_count}/{total} 指标看空")
        else:
            # 真正的冲突：趋势方向不明
            trend_direction = 'conflict'
            warnings.append(f"趋势冲突: {bullish_count}看多 vs {bearish_count}看空")
            return SignalAnalysis(
                signal_type=SignalType.CONFUSION,
                action=Action.WAIT,
                confidence_adjustment=-0.4,
                reasons=reasons,
                warnings=warnings
            )
        
        # 检查是否有金叉/死叉（趋势延续确认）
        if macd_cross_bullish and trend_direction == 'bullish':
            reasons.append("MACD金叉确认上涨趋势")
            return SignalAnalysis(
                signal_type=SignalType.TREND_CONTINUATION,
                action=Action.GO_LONG,
                confidence_adjustment=0.1,
                reasons=reasons,
                warnings=warnings
            )
        
        if macd_cross_bearish and trend_direction == 'bearish':
            reasons.append("MACD死叉确认下跌趋势")
            return SignalAnalysis(
                signal_type=SignalType.TREND_CONTINUATION,
                action=Action.GO_SHORT,
                confidence_adjustment=0.1,
                reasons=reasons,
                warnings=warnings
            )
        
        # 普通趋势延续
        if trend_direction == 'bullish':
            return SignalAnalysis(
                signal_type=SignalType.TREND_CONTINUATION,
                action=Action.GO_LONG,
                confidence_adjustment=0.05,
                reasons=reasons,
                warnings=warnings
            )
        else:
            return SignalAnalysis(
                signal_type=SignalType.TREND_CONTINUATION,
                action=Action.GO_SHORT,
                confidence_adjustment=0.05,
                reasons=reasons,
                warnings=warnings
            )
    
    def _analyze_momentum(self, indicators: Dict) -> SignalAnalysis:
        """
        动量类分析
        关键：区分超买超卖 vs 趋势动量
        """
        reasons = []
        warnings = []
        
        rsi = indicators['rsi']
        stoch = indicators['stochastic']
        cci = indicators['cci']['value']
        wr = indicators['williams_r']['value']
        mfi = indicators['mfi']['value']
        
        # 判断超买超卖
        oversold = rsi < 30 or stoch['k'] < 20 or cci < -100 or wr < -80 or mfi < 20
        overbought = rsi > 70 or stoch['k'] > 80 or cci > 100 or wr > -20 or mfi > 80
        
        # 统计超卖/超买指标数量
        oversold_count = sum([
            rsi < 30,
            stoch['k'] < 20,
            cci < -100,
            wr < -80,
            mfi < 20
        ])
        
        overbought_count = sum([
            rsi > 70,
            stoch['k'] > 80,
            cci > 100,
            wr > -20,
            mfi > 80
        ])
        
        # 场景1：严重超卖（多个指标）
        if oversold_count >= 3:
            reasons.append(f"严重超卖: RSI={rsi:.1f}, KDJ={stoch['k']:.1f}, CCI={cci:.1f}")
            warnings.append("超卖反弹概率高，但需等待企稳确认")
            return SignalAnalysis(
                signal_type=SignalType.REVERSAL,
                action=Action.GO_LONG,  # 反转做多
                confidence_adjustment=0.15,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景2：严重超买（多个指标）
        if overbought_count >= 3:
            reasons.append(f"严重超买: RSI={rsi:.1f}, KDJ={stoch['k']:.1f}, CCI={cci:.1f}")
            warnings.append("超买回调概率高，但需等待确认")
            return SignalAnalysis(
                signal_type=SignalType.REVERSAL,
                action=Action.GO_SHORT,  # 反转做空
                confidence_adjustment=0.15,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景3：轻度超卖/超买
        if oversold_count >= 1:
            reasons.append(f"轻度超卖: RSI={rsi:.1f}")
            return SignalAnalysis(
                signal_type=SignalType.REVERSAL,
                action=Action.WAIT,  # 等待确认
                confidence_adjustment=0.05,
                reasons=reasons,
                warnings=warnings
            )
        
        if overbought_count >= 1:
            reasons.append(f"轻度超买: RSI={rsi:.1f}")
            return SignalAnalysis(
                signal_type=SignalType.REVERSAL,
                action=Action.WAIT,
                confidence_adjustment=0.05,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景4：动量中性
        return SignalAnalysis(
            signal_type=SignalType.TREND_CONTINUATION,
            action=Action.NO_ACTION,
            confidence_adjustment=0.0,
            reasons=["动量中性"],
            warnings=warnings
        )
    
    def _analyze_volume(self, indicators: Dict) -> SignalAnalysis:
        """
        量价分析
        关键：检测背离
        """
        reasons = []
        warnings = []
        
        obv = indicators['obv']
        price = indicators['current_price']
        ema = indicators['ema']['value']
        
        # OBV 方向
        obv_rising = obv['value'] > obv['prev_value']
        price_above_ema = price > ema
        
        # 场景1：顶背离（价格涨 + OBV跌）
        if price_above_ema and not obv_rising:
            reasons.append("顶背离: 价格上涨但OBV下降")
            warnings.append("上涨动力不足，可能见顶")
            return SignalAnalysis(
                signal_type=SignalType.DIVERGENCE,
                action=Action.REDUCE_POSITION,
                confidence_adjustment=-0.2,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景2：底背离（价格跌 + OBV涨）
        if not price_above_ema and obv_rising:
            reasons.append("底背离: 价格下跌但OBV上升")
            warnings.append("下跌动力不足，可能见底")
            return SignalAnalysis(
                signal_type=SignalType.DIVERGENCE,
                action=Action.GO_LONG,  # 底背离做多
                confidence_adjustment=0.1,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景3：量价配合
        if price_above_ema and obv_rising:
            reasons.append("量价配合: 价涨量增")
            return SignalAnalysis(
                signal_type=SignalType.TREND_CONTINUATION,
                action=Action.NO_ACTION,
                confidence_adjustment=0.05,
                reasons=reasons,
                warnings=warnings
            )
        
        # 场景4：量价背离但不严重
        return SignalAnalysis(
            signal_type=SignalType.TREND_CONTINUATION,
            action=Action.NO_ACTION,
            confidence_adjustment=0.0,
            reasons=["量价中性"],
            warnings=warnings
        )
    
    def _combine_analyses(self, result: ConflictResult):
        """
        综合所有分析结果
        """
        all_reasons = []
        all_warnings = []
        adjustments = []
        
        # 收集所有分析
        analyses = [
            result.trend_analysis,
            result.momentum_analysis,
            result.volume_analysis,
        ]
        
        for analysis in analyses:
            if analysis:
                all_reasons.extend(analysis.reasons)
                all_warnings.extend(analysis.warnings)
                adjustments.append(analysis.confidence_adjustment)
        
        # 计算综合调整
        # 趋势分析权重最高
        if result.trend_analysis:
            trend_weight = 0.5
            momentum_weight = 0.3
            volume_weight = 0.2
        else:
            trend_weight = 0.33
            momentum_weight = 0.33
            volume_weight = 0.33
        
        weighted_adjustment = 0
        if result.trend_analysis:
            weighted_adjustment += result.trend_analysis.confidence_adjustment * trend_weight
        if result.momentum_analysis:
            weighted_adjustment += result.momentum_analysis.confidence_adjustment * momentum_weight
        if result.volume_analysis:
            weighted_adjustment += result.volume_analysis.confidence_adjustment * volume_weight
        
        # 确定最终动作
        # 优先级：趋势 > 动量 > 量价
        if result.trend_analysis and result.trend_analysis.action != Action.NO_ACTION:
            final_action = result.trend_analysis.action
        elif result.momentum_analysis and result.momentum_analysis.action != Action.NO_ACTION:
            final_action = result.momentum_analysis.action
        elif result.volume_analysis and result.volume_analysis.action != Action.NO_ACTION:
            final_action = result.volume_analysis.action
        else:
            final_action = Action.NO_ACTION
        
        # 检查是否有冲突
        actions = []
        for a in analyses:
            if a and a.action not in [Action.NO_ACTION, Action.WAIT]:
                actions.append(a.action)
        
        # 如果有相反的动作，降低置信度
        if Action.GO_LONG in actions and Action.GO_SHORT in actions:
            weighted_adjustment -= 0.2
            all_warnings.append("信号冲突: 有指标看多，有指标看空")
        
        result.final_adjustment = max(-0.5, min(0.3, weighted_adjustment))
        result.final_action = final_action
        result.all_reasons = all_reasons
        result.all_warnings = all_warnings


def detect_conflicts(indicators: Dict) -> ConflictResult:
    """
    便捷函数：检测冲突
    """
    detector = ConflictDetector()
    return detector.analyze(indicators)


if __name__ == "__main__":
    print("Conflict Detector v2 Module Loaded")
