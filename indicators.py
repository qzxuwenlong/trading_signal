"""
技术指标计算模块
包含：阻力位、支撑位、布林带、趋势线、谐波形态、RSI
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class SupportResistanceLevel:
    """支撑阻力位"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 强度 0-1
    touches: int     # 触碰次数
    index: int       # 位置索引


@dataclass
class BollingerBands:
    """布林带"""
    upper: float
    middle: float
    lower: float
    bandwidth: float
    percent_b: float  # %B 指标


@dataclass
class TrendLine:
    """趋势线"""
    slope: float
    intercept: float
    trend_type: str  # 'up', 'down', 'sideways'
    strength: float
    points: List[Tuple[int, float]]


@dataclass
class HarmonicPattern:
    """谐波形态"""
    pattern_type: str  # 'Gartley', 'Butterfly', 'Bat', 'Crab', 'Cypher'
    points: Dict[str, Tuple[int, float]]  # X, A, B, C, D 点
    completion: float  # 完成度 0-1
    direction: str     # 'bullish' or 'bearish'


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化
        
        Args:
            df: K线数据，包含 timestamps, open, high, low, close, volume
        """
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.open = df['open'].values
        
    def calculate_bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> BollingerBands:
        """
        计算布林带
        
        Args:
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            布林带数据
        """
        close_series = pd.Series(self.close)
        
        middle = close_series.rolling(window=period).mean().iloc[-1]
        std = close_series.rolling(window=period).std().iloc[-1]
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        bandwidth = (upper - lower) / middle
        
        # %B 指标：当前价格在布林带中的位置
        current_price = self.close[-1]
        percent_b = (current_price - lower) / (upper - lower)
        
        return BollingerBands(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )
    
    def calculate_rsi(self, period: int = 14) -> float:
        """
        计算 RSI
        
        Args:
            period: 周期
            
        Returns:
            RSI 值
        """
        close_series = pd.Series(self.close)
        
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def find_support_resistance(
        self,
        lookback: int = 100,
        threshold: float = 0.02
    ) -> List[SupportResistanceLevel]:
        """
        查找支撑阻力位
        
        Args:
            lookback: 回看周期
            threshold: 价格聚类阈值
            
        Returns:
            支撑阻力位列表
        """
        # 使用最近 lookback 根K线
        recent_high = self.high[-lookback:]
        recent_low = self.low[-lookback:]
        
        # 收集所有极值点
        levels = []
        
        # 找局部高点和低点
        for i in range(2, len(recent_high) - 2):
            # 局部高点
            if (recent_high[i] > recent_high[i-1] and 
                recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and 
                recent_high[i] > recent_high[i+2]):
                levels.append((recent_high[i], 'resistance'))
            
            # 局部低点
            if (recent_low[i] < recent_low[i-1] and 
                recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and 
                recent_low[i] < recent_low[i+2]):
                levels.append((recent_low[i], 'support'))
        
        if not levels:
            return []
        
        # 聚类相似的价格
        levels.sort(key=lambda x: x[0])
        clustered = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            if (levels[i][0] - current_cluster[-1][0]) / current_cluster[-1][0] < threshold:
                current_cluster.append(levels[i])
            else:
                # 计算聚类的平均价格和强度
                avg_price = np.mean([l[0] for l in current_cluster])
                touches = len(current_cluster)
                strength = min(touches / 5, 1.0)  # 最多5次触碰为最强
                level_type = max(set([l[1] for l in current_cluster]), key=[l[1] for l in current_cluster].count)
                
                clustered.append(SupportResistanceLevel(
                    price=avg_price,
                    level_type=level_type,
                    strength=strength,
                    touches=touches,
                    index=0
                ))
                current_cluster = [levels[i]]
        
        # 处理最后一个聚类
        if current_cluster:
            avg_price = np.mean([l[0] for l in current_cluster])
            touches = len(current_cluster)
            strength = min(touches / 5, 1.0)
            level_type = max(set([l[1] for l in current_cluster]), key=[l[1] for l in current_cluster].count)
            
            clustered.append(SupportResistanceLevel(
                price=avg_price,
                level_type=level_type,
                strength=strength,
                touches=touches,
                index=0
            ))
        
        return clustered
    
    def calculate_trendline(
        self,
        lookback: int = 50
    ) -> TrendLine:
        """
        计算趋势线
        
        Args:
            lookback: 回看周期
            
        Returns:
            趋势线数据
        """
        close_series = self.close[-lookback:]
        x = np.arange(len(close_series))
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_series)
        
        # 判断趋势类型
        if slope > 0.001:
            trend_type = 'up'
        elif slope < -0.001:
            trend_type = 'down'
        else:
            trend_type = 'sideways'
        
        # 计算强度 (基于 R²)
        strength = r_value ** 2
        
        # 找到趋势线上的关键点
        points = []
        for i in range(0, len(close_series), len(close_series) // 5):
            price_at_i = slope * i + intercept
            points.append((i, price_at_i))
        
        return TrendLine(
            slope=slope,
            intercept=intercept,
            trend_type=trend_type,
            strength=strength,
            points=points
        )
    
    def find_harmonic_patterns(
        self,
        min_swing: float = 0.05
    ) -> List[HarmonicPattern]:
        """
        查找谐波形态
        
        Args:
            min_swing: 最小摆动幅度
            
        Returns:
            谐波形态列表
        """
        patterns = []
        
        # 简化的谐波形态检测
        # 使用最近的价格数据
        lookback = min(100, len(self.close))
        close = self.close[-lookback:]
        high = self.high[-lookback:]
        low = self.low[-lookback:]
        
        # 找到摆动点
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(close) - 2):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                swing_highs.append((i, high[i]))
            if low[i] < low[i-1] and low[i] < low[i+1]:
                swing_lows.append((i, low[i]))
        
        # 检测 Gartley 形态 (简化版)
        if len(swing_lows) >= 2 and len(swing_highs) >= 2:
            # 检查是否有 XABCD 结构
            all_points = sorted(swing_lows + swing_highs, key=lambda x: x[0])
            
            if len(all_points) >= 5:
                # 取最后5个点作为潜在的谐波形态
                points_dict = {
                    'X': all_points[-5],
                    'A': all_points[-4],
                    'B': all_points[-3],
                    'C': all_points[-2],
                    'D': all_points[-1]
                }
                
                # 计算 Fibonacci 回撤
                xa = abs(points_dict['A'][1] - points_dict['X'][1])
                ab = abs(points_dict['B'][1] - points_dict['A'][1])
                bc = abs(points_dict['C'][1] - points_dict['B'][1])
                cd = abs(points_dict['D'][1] - points_dict['C'][1])
                
                if xa > 0:
                    ab_ratio = ab / xa
                    bc_ratio = bc / ab if ab > 0 else 0
                    cd_ratio = cd / bc if bc > 0 else 0
                    
                    # Gartley: AB=0.618XA, BC=0.382-0.886AB, CD=1.27-1.618BC
                    if (0.5 < ab_ratio < 0.7 and 
                        0.3 < bc_ratio < 0.9 and
                        1.2 < cd_ratio < 1.7):
                        
                        direction = 'bullish' if points_dict['D'][1] < points_dict['B'][1] else 'bearish'
                        
                        patterns.append(HarmonicPattern(
                            pattern_type='Gartley',
                            points=points_dict,
                            completion=0.8,
                            direction=direction
                        ))
        
        return patterns
    
    def calculate_all_indicators(self) -> Dict:
        """
        计算所有技术指标
        
        Returns:
            所有指标的字典
        """
        # 布林带
        bollinger = self.calculate_bollinger_bands()
        
        # RSI
        rsi = self.calculate_rsi()
        
        # 支撑阻力
        sr_levels = self.find_support_resistance()
        
        # 趋势线
        trendline = self.calculate_trendline()
        
        # 谐波形态
        harmonic = self.find_harmonic_patterns()
        
        # 当前价格
        current_price = self.close[-1]
        
        return {
            'current_price': current_price,
            'bollinger': {
                'upper': bollinger.upper,
                'middle': bollinger.middle,
                'lower': bollinger.lower,
                'bandwidth': bollinger.bandwidth,
                'percent_b': bollinger.percent_b,
            },
            'rsi': rsi,
            'support_resistance': [
                {
                    'price': level.price,
                    'type': level.level_type,
                    'strength': level.strength,
                    'touches': level.touches,
                }
                for level in sr_levels
            ],
            'trendline': {
                'slope': trendline.slope,
                'type': trendline.trend_type,
                'strength': trendline.strength,
            },
            'harmonic': [
                {
                    'type': pattern.pattern_type,
                    'direction': pattern.direction,
                    'completion': pattern.completion,
                }
                for pattern in harmonic
            ],
        }
    
    def get_key_levels(self) -> Dict:
        """
        获取关键交易位
        
        Returns:
            关键位信息
        """
        indicators = self.calculate_all_indicators()
        current_price = indicators['current_price']
        
        # 找到最近的支撑和阻力
        supports = [l for l in indicators['support_resistance'] if l['type'] == 'support']
        resistances = [l for l in indicators['support_resistance'] if l['type'] == 'resistance']
        
        # 排序
        supports.sort(key=lambda x: x['price'], reverse=True)
        resistances.sort(key=lambda x: x['price'])
        
        # 找最近的支撑和阻力
        nearest_support = None
        nearest_resistance = None
        
        for s in supports:
            if s['price'] < current_price:
                nearest_support = s
                break
        
        for r in resistances:
            if r['price'] > current_price:
                nearest_resistance = r
                break
        
        return {
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'bollinger_position': indicators['bollinger'],
            'rsi': indicators['rsi'],
            'trend': indicators['trendline'],
            'harmonic': indicators['harmonic'],
        }


def analyze_technicals(df: pd.DataFrame) -> Dict:
    """
    便捷函数：分析技术指标
    
    Args:
        df: K线数据
        
    Returns:
        技术分析结果
    """
    analyzer = TechnicalIndicators(df)
    return analyzer.get_key_levels()


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, ".")
    from okx_data import get_market_data
    
    print("获取 BTC-USDT 数据...")
    df = get_market_data("BTC-USDT", "15m", 200)
    
    print("分析技术指标...")
    result = analyze_technicals(df)
    
    print(f"\n当前价格: ${result['current_price']:.2f}")
    
    if result['nearest_support']:
        print(f"最近支撑: ${result['nearest_support']['price']:.2f}")
    
    if result['nearest_resistance']:
        print(f"最近阻力: ${result['nearest_resistance']:.2f}")
    
    print(f"RSI: {result['rsi']:.2f}")
    print(f"趋势: {result['trend']['type']}")
    print(f"布林带 %B: {result['bollinger_position']['percent_b']:.2f}")
