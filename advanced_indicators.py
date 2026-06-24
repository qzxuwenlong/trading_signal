"""
技术指标计算模块（完整版）
包含：趋势、动量、波动、量能、形态识别
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy import stats
import math


@dataclass
class EMA:
    """EMA指标"""
    value: float
    prev: float


@dataclass
class MACD:
    """MACD指标"""
    macd: float
    signal: float
    histogram: float
    prev_histogram: float


@dataclass
class ADX:
    """ADX指标"""
    adx: float
    plus_di: float
    minus_di: float


@dataclass
class Ichimoku:
    """一目均衡指标"""
    tenkan: float  # 转换线
    kijun: float   # 基准线
    senkou_a: float  # 先行带A
    senkou_b: float  # 先行带B
    chikou: float    # 迟行带


@dataclass
class ParabolicSAR:
    """抛物线SAR"""
    value: float
    direction: str  # 'long' or 'short'


@dataclass
class Stochastic:
    """随机指标 (KDJ)"""
    k: float
    d: float
    j: float


@dataclass
class CCI:
    """商品通道指数"""
    value: float


@dataclass
class WilliamsR:
    """威廉指标"""
    value: float


@dataclass
class MFI:
    """资金流量指数"""
    value: float


@dataclass
class ROC:
    """变动率"""
    value: float


@dataclass
class ATR:
    """真实波幅"""
    value: float


@dataclass
class KeltnerChannel:
    """肯特纳通道"""
    upper: float
    middle: float
    lower: float


@dataclass
class DonchianChannel:
    """唐奇安通道"""
    upper: float
    middle: float
    lower: float


@dataclass
class OBV:
    """能量潮"""
    value: float
    prev_value: float


@dataclass
class VWAP:
    """成交量加权均价"""
    value: float


@dataclass
class VolumeProfile:
    """成交量分布"""
    poc: float  # 控制点（高量区价格）
    value_area_high: float
    value_area_low: float


@dataclass
class FibonacciLevels:
    """斐波那契回撤"""
    levels: Dict[float, float]  # ratio -> price


@dataclass
class PivotPoints:
    """枢轴点"""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


@dataclass
class CandlestickPattern:
    """K线形态"""
    name: str
    type: str  # 'bullish' or 'bearish'
    strength: float  # 0-1


class AdvancedIndicators:
    """高级技术指标计算器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化
        
        Args:
            df: K线数据，包含 timestamps, open, high, low, close, volume
        """
        self.df = df.copy()
        self.close = df['close'].values.astype(float)
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.open = df['open'].values.astype(float)
        self.volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(len(df))
    
    # ==================== 趋势类指标 ====================
    
    def calculate_ema(self, period: int = 20) -> EMA:
        """
        计算 EMA (指数移动平均)
        """
        close_series = pd.Series(self.close)
        ema = close_series.ewm(span=period, adjust=False).mean()
        return EMA(
            value=ema.iloc[-1],
            prev=ema.iloc[-2] if len(ema) > 1 else ema.iloc[-1]
        )
    
    def calculate_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> MACD:
        """
        计算 MACD
        """
        close_series = pd.Series(self.close)
        
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return MACD(
            macd=macd_line.iloc[-1],
            signal=signal_line.iloc[-1],
            histogram=histogram.iloc[-1],
            prev_histogram=histogram.iloc[-2] if len(histogram) > 1 else 0
        )
    
    def calculate_adx(self, period: int = 14) -> ADX:
        """
        计算 ADX (平均趋向指数)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return ADX(
            adx=adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0,
            plus_di=plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0,
            minus_di=minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0
        )
    
    def calculate_ichimoku(self) -> Ichimoku:
        """
        计算一目均衡 (Ichimoku Cloud)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        # 转换线 (9周期)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # 基准线 (26周期)
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # 先行带A
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # 先行带B (52周期)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # 迟行带
        chikou = close.shift(-26)
        
        return Ichimoku(
            tenkan=tenkan.iloc[-1] if not np.isnan(tenkan.iloc[-1]) else self.close[-1],
            kijun=kijun.iloc[-1] if not np.isnan(kijun.iloc[-1]) else self.close[-1],
            senkou_a=senkou_a.iloc[-1] if not np.isnan(senkou_a.iloc[-1]) else self.close[-1],
            senkou_b=senkou_b.iloc[-1] if not np.isnan(senkou_b.iloc[-1]) else self.close[-1],
            chikou=chikou.iloc[-1] if not np.isnan(chikou.iloc[-1]) else self.close[-1]
        )
    
    def calculate_parabolic_sar(self, af_start=0.02, af_step=0.02, af_max=0.2) -> ParabolicSAR:
        """
        计算抛物线 SAR
        """
        high = self.high
        low = self.low
        close = self.close
        
        length = len(high)
        sar = np.zeros(length)
        af = af_start
        ep = high[0]
        trend = 1  # 1 = up, -1 = down
        
        sar[0] = low[0]
        
        for i in range(1, length):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                
                if low[i] < sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)
        
        direction = 'long' if close[-1] > sar[-1] else 'short'
        
        return ParabolicSAR(value=sar[-1], direction=direction)
    
    # ==================== 动量类指标 ====================
    
    def calculate_stochastic(self, k_period=14, d_period=3, smooth=3) -> Stochastic:
        """
        计算随机指标 (KDJ)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        k = rsv.ewm(com=smooth-1, adjust=False).mean()
        d = k.ewm(com=d_period-1, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return Stochastic(
            k=k.iloc[-1] if not np.isnan(k.iloc[-1]) else 50,
            d=d.iloc[-1] if not np.isnan(d.iloc[-1]) else 50,
            j=j.iloc[-1] if not np.isnan(j.iloc[-1]) else 50
        )
    
    def calculate_cci(self, period: int = 20) -> CCI:
        """
        计算商品通道指数 (CCI)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (tp - sma) / (0.015 * mad)
        
        return CCI(value=cci.iloc[-1] if not np.isnan(cci.iloc[-1]) else 0)
    
    def calculate_williams_r(self, period: int = 14) -> WilliamsR:
        """
        计算威廉指标 (%R)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return WilliamsR(value=wr.iloc[-1] if not np.isnan(wr.iloc[-1]) else -50)
    
    def calculate_mfi(self, period: int = 14) -> MFI:
        """
        计算资金流量指数 (MFI)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        volume = pd.Series(self.volume)
        
        tp = (high + low + close) / 3
        mf = tp * volume
        
        positive_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0))
        negative_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0))
        
        positive_sum = positive_mf.rolling(window=period).sum()
        negative_sum = negative_mf.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_sum / negative_sum))
        
        return MFI(value=mfi.iloc[-1] if not np.isnan(mfi.iloc[-1]) else 50)
    
    def calculate_roc(self, period: int = 12) -> ROC:
        """
        计算变动率 (ROC)
        """
        close = pd.Series(self.close)
        roc = (close - close.shift(period)) / close.shift(period) * 100
        
        return ROC(value=roc.iloc[-1] if not np.isnan(roc.iloc[-1]) else 0)
    
    # ==================== 波动类指标 ====================
    
    def calculate_atr(self, period: int = 14) -> ATR:
        """
        计算真实波幅 (ATR)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        return ATR(value=atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0)
    
    def calculate_keltner_channel(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> KeltnerChannel:
        """
        计算肯特纳通道
        """
        close = pd.Series(self.close)
        ema = close.ewm(span=ema_period, adjust=False).mean()
        
        atr = self.calculate_atr(atr_period)
        
        upper = ema + multiplier * atr.value
        lower = ema - multiplier * atr.value
        
        return KeltnerChannel(
            upper=upper.iloc[-1],
            middle=ema.iloc[-1],
            lower=lower.iloc[-1]
        )
    
    def calculate_donchian_channel(self, period: int = 20) -> DonchianChannel:
        """
        计算唐奇安通道
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return DonchianChannel(
            upper=upper.iloc[-1],
            middle=middle.iloc[-1],
            lower=lower.iloc[-1]
        )
    
    # ==================== 量能类指标 ====================
    
    def calculate_obv(self) -> OBV:
        """
        计算能量潮 (OBV)
        """
        close = pd.Series(self.close)
        volume = pd.Series(self.volume)
        
        obv = pd.Series(np.where(
            close > close.shift(1),
            volume,
            np.where(close < close.shift(1), -volume, 0)
        )).cumsum()
        
        return OBV(
            value=obv.iloc[-1],
            prev_value=obv.iloc[-2] if len(obv) > 1 else obv.iloc[-1]
        )
    
    def calculate_vwap(self) -> VWAP:
        """
        计算成交量加权均价 (VWAP)
        """
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        volume = pd.Series(self.volume)
        
        tp = (high + low + close) / 3
        vwap = (tp * volume).cumsum() / volume.cumsum()
        
        return VWAP(value=vwap.iloc[-1] if not np.isnan(vwap.iloc[-1]) else self.close[-1])
    
    def calculate_volume_profile(self, bins: int = 20) -> VolumeProfile:
        """
        计算成交量分布
        """
        price_range = np.linspace(self.low.min(), self.high.max(), bins)
        volume_profile = np.zeros(bins)
        
        for i in range(len(self.close)):
            for j in range(len(price_range) - 1):
                if price_range[j] <= self.close[i] < price_range[j + 1]:
                    volume_profile[j] += self.volume[i]
                    break
        
        poc_idx = np.argmax(volume_profile)
        poc = (price_range[poc_idx] + price_range[poc_idx + 1]) / 2
        
        # Value Area (70% volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative += volume_profile[idx]
            value_area_indices.append(idx)
            if cumulative >= target_volume:
                break
        
        va_high = price_range[max(value_area_indices) + 1]
        va_low = price_range[min(value_area_indices)]
        
        return VolumeProfile(
            poc=poc,
            value_area_high=va_high,
            value_area_low=va_low
        )
    
    # ==================== 形态识别 ====================
    
    def calculate_fibonacci_levels(
        self,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
        lookback: int = 100
    ) -> FibonacciLevels:
        """
        计算斐波那契回撤位
        """
        if high_price is None:
            high_price = np.max(self.high[-lookback:])
        if low_price is None:
            low_price = np.min(self.low[-lookback:])
        
        diff = high_price - low_price
        
        levels = {
            0.0: high_price,
            0.236: high_price - diff * 0.236,
            0.382: high_price - diff * 0.382,
            0.5: high_price - diff * 0.5,
            0.618: high_price - diff * 0.618,
            0.786: high_price - diff * 0.786,
            1.0: low_price,
            1.272: low_price - diff * 0.272,
            1.618: low_price - diff * 0.618,
        }
        
        return FibonacciLevels(levels=levels)
    
    def calculate_pivot_points(self) -> PivotPoints:
        """
        计算枢轴点
        """
        high = self.high[-1]
        low = self.low[-1]
        close = self.close[-1]
        
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return PivotPoints(
            pivot=pivot,
            r1=r1, r2=r2, r3=r3,
            s1=s1, s2=s2, s3=s3
        )
    
    def detect_candlestick_patterns(self) -> List[CandlestickPattern]:
        """
        检测K线形态
        """
        patterns = []
        
        # 确保有足够的数据
        if len(self.close) < 3:
            return patterns
        
        # 获取最近3根K线
        o1, h1, l1, c1 = self.open[-3], self.high[-3], self.low[-3], self.close[-3]
        o2, h2, l2, c2 = self.open[-2], self.high[-2], self.low[-2], self.close[-2]
        o3, h3, l3, c3 = self.open[-1], self.high[-1], self.low[-1], self.close[-1]
        
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        
        upper_shadow1 = h1 - max(o1, c1)
        lower_shadow1 = min(o1, c1) - l1
        upper_shadow2 = h2 - max(o2, c2)
        lower_shadow2 = min(o2, c2) - l2
        upper_shadow3 = h3 - max(o3, c3)
        lower_shadow3 = min(o3, c3) - l3
        
        total_range1 = h1 - l1
        total_range2 = h2 - l2
        total_range3 = h3 - l3
        
        # 1. 锤子线 (Hammer) - 看涨
        if (lower_shadow3 > 2 * body3 and 
            upper_shadow3 < body3 * 0.3 and
            total_range3 > 0):
            patterns.append(CandlestickPattern(
                name='Hammer',
                type='bullish',
                strength=min(lower_shadow3 / body3 / 3, 1.0)
            ))
        
        # 2. 倒锤子线 (Inverted Hammer) - 看涨
        if (upper_shadow3 > 2 * body3 and 
            lower_shadow3 < body3 * 0.3 and
            total_range3 > 0):
            patterns.append(CandlestickPattern(
                name='Inverted Hammer',
                type='bullish',
                strength=min(upper_shadow3 / body3 / 3, 1.0)
            ))
        
        # 3. 吞没形态 (Engulfing)
        if c2 < o2 and c3 > o3 and c3 > o2 and o3 < c2:  # 看涨吞没
            patterns.append(CandlestickPattern(
                name='Bullish Engulfing',
                type='bullish',
                strength=min(body3 / body2, 1.0) if body2 > 0 else 0.5
            ))
        elif c2 > o2 and c3 < o3 and c3 < o2 and o3 > c2:  # 看跌吞没
            patterns.append(CandlestickPattern(
                name='Bearish Engulfing',
                type='bearish',
                strength=min(body3 / body2, 1.0) if body2 > 0 else 0.5
            ))
        
        # 4. 十字星 (Doji)
        if total_range3 > 0 and body3 < total_range3 * 0.1:
            patterns.append(CandlestickPattern(
                name='Doji',
                type='bullish' if c2 < o2 else 'bearish',
                strength=0.5
            ))
        
        # 5. 早晨之星 (Morning Star) - 看涨
        if (c1 < o1 and body1 > 0 and
            body2 < body1 * 0.3 and
            c3 > o3 and c3 > (o1 + c1) / 2):
            patterns.append(CandlestickPattern(
                name='Morning Star',
                type='bullish',
                strength=0.8
            ))
        
        # 6. 黄昏之星 (Evening Star) - 看跌
        if (c1 > o1 and body1 > 0 and
            body2 < body1 * 0.3 and
            c3 < o3 and c3 < (o1 + c1) / 2):
            patterns.append(CandlestickPattern(
                name='Evening Star',
                type='bearish',
                strength=0.8
            ))
        
        # 7. 三只乌鸦 (Three Black Crows) - 看跌
        if (c1 < o1 and c2 < o2 and c3 < o3 and
            o2 < c1 and o3 < c2 and
            body1 > 0 and body2 > 0 and body3 > 0):
            patterns.append(CandlestickPattern(
                name='Three Black Crows',
                type='bearish',
                strength=0.9
            ))
        
        # 8. 三个白兵 (Three White Soldiers) - 看涨
        if (c1 > o1 and c2 > o2 and c3 > o3 and
            o2 > c1 and o3 > c2 and
            body1 > 0 and body2 > 0 and body3 > 0):
            patterns.append(CandlestickPattern(
                name='Three White Soldiers',
                type='bullish',
                strength=0.9
            ))
        
        # 9. 孕线 (Harami)
        if abs(c1 - o1) > abs(c2 - o2) * 2:
            if c1 < o1 and c2 > o2 and o2 > c1 and c2 < o1:  # 看涨孕线
                patterns.append(CandlestickPattern(
                    name='Bullish Harami',
                    type='bullish',
                    strength=0.6
                ))
            elif c1 > o1 and c2 < o2 and o2 < c1 and c2 > o1:  # 看跌孕线
                patterns.append(CandlestickPattern(
                    name='Bearish Harami',
                    type='bearish',
                    strength=0.6
                ))
        
        return patterns
    
    # ==================== 支撑阻力（原有）====================
    
    def find_support_resistance(
        self,
        lookback: int = 100,
        threshold: float = 0.02
    ) -> List[Dict]:
        """
        查找支撑阻力位
        """
        recent_high = self.high[-lookback:]
        recent_low = self.low[-lookback:]
        
        levels = []
        
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and 
                recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and 
                recent_high[i] > recent_high[i+2]):
                levels.append({'price': recent_high[i], 'type': 'resistance'})
            
            if (recent_low[i] < recent_low[i-1] and 
                recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and 
                recent_low[i] < recent_low[i+2]):
                levels.append({'price': recent_low[i], 'type': 'support'})
        
        if not levels:
            return []
        
        levels.sort(key=lambda x: x['price'])
        clustered = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            if (levels[i]['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] < threshold:
                current_cluster.append(levels[i])
            else:
                avg_price = np.mean([l['price'] for l in current_cluster])
                touches = len(current_cluster)
                strength = min(touches / 5, 1.0)
                level_type = max(set([l['type'] for l in current_cluster]), 
                               key=[l['type'] for l in current_cluster].count)
                
                clustered.append({
                    'price': avg_price,
                    'type': level_type,
                    'strength': strength,
                    'touches': touches
                })
                current_cluster = [levels[i]]
        
        if current_cluster:
            avg_price = np.mean([l['price'] for l in current_cluster])
            touches = len(current_cluster)
            strength = min(touches / 5, 1.0)
            level_type = max(set([l['type'] for l in current_cluster]), 
                           key=[l['type'] for l in current_cluster].count)
            
            clustered.append({
                'price': avg_price,
                'type': level_type,
                'strength': strength,
                'touches': touches
            })
        
        return clustered
    
    # ==================== 布林带（原有）====================
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict:
        """
        计算布林带
        """
        close_series = pd.Series(self.close)
        
        middle = close_series.rolling(window=period).mean().iloc[-1]
        std = close_series.rolling(window=period).std().iloc[-1]
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        bandwidth = (upper - lower) / middle
        percent_b = (self.close[-1] - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    # ==================== RSI（原有）====================
    
    def calculate_rsi(self, period: int = 14) -> float:
        """
        计算 RSI
        """
        close_series = pd.Series(self.close)
        
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    
    # ==================== 趋势线（原有）====================
    
    def calculate_trendline(self, lookback: int = 50) -> Dict:
        """
        计算趋势线
        """
        close_series = self.close[-lookback:]
        x = np.arange(len(close_series))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_series)
        
        if slope > 0.001:
            trend_type = 'up'
        elif slope < -0.001:
            trend_type = 'down'
        else:
            trend_type = 'sideways'
        
        strength = r_value ** 2
        
        return {
            'slope': slope,
            'intercept': intercept,
            'type': trend_type,
            'strength': strength
        }
    
    # ==================== 综合方法 ====================
    
    def calculate_all_advanced_indicators(self) -> Dict[str, Any]:
        """
        计算所有高级技术指标
        """
        # 趋势类
        ema = self.calculate_ema()
        macd = self.calculate_macd()
        adx = self.calculate_adx()
        ichimoku = self.calculate_ichimoku()
        parabolic_sar = self.calculate_parabolic_sar()
        
        # 动量类
        stochastic = self.calculate_stochastic()
        cci = self.calculate_cci()
        williams_r = self.calculate_williams_r()
        mfi = self.calculate_mfi()
        roc = self.calculate_roc()
        
        # 波动类
        atr = self.calculate_atr()
        keltner = self.calculate_keltner_channel()
        donchian = self.calculate_donchian_channel()
        
        # 量能类
        obv = self.calculate_obv()
        vwap = self.calculate_vwap()
        volume_profile = self.calculate_volume_profile()
        
        # 形态
        fibonacci = self.calculate_fibonacci_levels()
        pivot = self.calculate_pivot_points()
        candlestick = self.detect_candlestick_patterns()
        
        # 原有指标
        bollinger = self.calculate_bollinger_bands()
        rsi = self.calculate_rsi()
        sr_levels = self.find_support_resistance()
        trendline = self.calculate_trendline()
        
        return {
            'current_price': self.close[-1],
            
            # 趋势类
            'ema': {'value': ema.value, 'prev': ema.prev},
            'macd': {
                'macd': macd.macd,
                'signal': macd.signal,
                'histogram': macd.histogram,
                'prev_histogram': macd.prev_histogram
            },
            'adx': {
                'adx': adx.adx,
                'plus_di': adx.plus_di,
                'minus_di': adx.minus_di
            },
            'ichimoku': {
                'tenkan': ichimoku.tenkan,
                'kijun': ichimoku.kijun,
                'senkou_a': ichimoku.senkou_a,
                'senkou_b': ichimoku.senkou_b,
                'chikou': ichimoku.chikou
            },
            'parabolic_sar': {
                'value': parabolic_sar.value,
                'direction': parabolic_sar.direction
            },
            
            # 动量类
            'stochastic': {
                'k': stochastic.k,
                'd': stochastic.d,
                'j': stochastic.j
            },
            'cci': {'value': cci.value},
            'williams_r': {'value': williams_r.value},
            'mfi': {'value': mfi.value},
            'roc': {'value': roc.value},
            
            # 波动类
            'atr': {'value': atr.value},
            'keltner': {
                'upper': keltner.upper,
                'middle': keltner.middle,
                'lower': keltner.lower
            },
            'donchian': {
                'upper': donchian.upper,
                'middle': donchian.middle,
                'lower': donchian.lower
            },
            
            # 量能类
            'obv': {'value': obv.value, 'prev_value': obv.prev_value},
            'vwap': {'value': vwap.value},
            'volume_profile': {
                'poc': volume_profile.poc,
                'value_area_high': volume_profile.value_area_high,
                'value_area_low': volume_profile.value_area_low
            },
            
            # 形态
            'fibonacci': fibonacci.levels,
            'pivot': {
                'pivot': pivot.pivot,
                'r1': pivot.r1, 'r2': pivot.r2, 'r3': pivot.r3,
                's1': pivot.s1, 's2': pivot.s2, 's3': pivot.s3
            },
            'candlestick_patterns': [
                {'name': p.name, 'type': p.type, 'strength': p.strength}
                for p in candlestick
            ],
            
            # 原有指标
            'bollinger': bollinger,
            'rsi': rsi,
            'support_resistance': sr_levels,
            'trendline': trendline,
        }


def analyze_all_indicators(df: pd.DataFrame) -> Dict:
    """
    便捷函数：分析所有技术指标
    
    Args:
        df: K线数据
        
    Returns:
        所有技术指标
    """
    analyzer = AdvancedIndicators(df)
    return analyzer.calculate_all_advanced_indicators()


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, ".")
    from okx_data import get_market_data
    
    print("获取 BTC-USDT 数据...")
    df = get_market_data("BTC-USDT", "15m", 200)
    
    if not df.empty:
        print("分析所有技术指标...")
        result = analyze_all_indicators(df)
        
        print(f"\n当前价格: ${result['current_price']:.2f}")
        print(f"\n趋势类:")
        print(f"  EMA: ${result['ema']['value']:.2f}")
        print(f"  MACD: {result['macd']['macd']:.4f}")
        print(f"  ADX: {result['adx']['adx']:.2f}")
        print(f"  Parabolic SAR: ${result['parabolic_sar']['value']:.2f} ({result['parabolic_sar']['direction']})")
        
        print(f"\n动量类:")
        print(f"  RSI: {result['rsi']:.2f}")
        print(f"  Stochastic K: {result['stochastic']['k']:.2f}")
        print(f"  CCI: {result['cci']['value']:.2f}")
        print(f"  Williams %R: {result['williams_r']['value']:.2f}")
        print(f"  MFI: {result['mfi']['value']:.2f}")
        print(f"  ROC: {result['roc']['value']:.2f}%")
        
        print(f"\n波动类:")
        print(f"  ATR: ${result['atr']['value']:.2f}")
        print(f"  Keltner: ${result['keltner']['lower']:.2f} - ${result['keltner']['upper']:.2f}")
        print(f"  Donchian: ${result['donchian']['lower']:.2f} - ${result['donchian']['upper']:.2f}")
        
        print(f"\n量能类:")
        print(f"  OBV: {result['obv']['value']:.0f}")
        print(f"  VWAP: ${result['vwap']['value']:.2f}")
        
        print(f"\n形态识别:")
        for pattern in result['candlestick_patterns']:
            print(f"  {pattern['name']}: {pattern['type']} (strength: {pattern['strength']:.2f})")
    else:
        print("获取数据失败")
