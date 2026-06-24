"""
Kronos 模型调用模块
封装 Kronos 模型的预测功能
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model not available")


@dataclass
class KronosPrediction:
    """Kronos 预测结果"""
    timestamps: List
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: Optional[np.ndarray] = None
    trend: str = ""  # 'up', 'down', 'sideways'
    change_pct: float = 0.0


class KronosPredictorWrapper:
    """Kronos 预测器封装"""
    
    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        max_context: int = 512
    ):
        """
        初始化
        
        Args:
            model_name: 模型名称
            tokenizer_name: 分词器名称
            device: 设备 (cpu/cuda)
            max_context: 最大上下文长度
        """
        if not MODEL_AVAILABLE:
            raise ImportError("Kronos model not available")
        
        self.device = device
        self.max_context = max_context
        
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        
        print(f"Loading model: {model_name}...")
        self.model = Kronos.from_pretrained(model_name)
        
        print("Initializing predictor...")
        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=device,
            max_context=max_context
        )
        
        print("Kronos model loaded successfully!")
    
    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int = 6,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
    ) -> KronosPrediction:
        """
        预测未来K线
        
        Args:
            df: 历史K线数据
            pred_len: 预测长度
            T: 温度参数
            top_p: nucleus sampling 概率
            sample_count: 采样数量
            
        Returns:
            预测结果
        """
        # 准备输入数据
        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_cols.append('volume')
        
        x_df = df[required_cols].tail(self.max_context)
        x_timestamp = df['timestamps'].tail(self.max_context)
        
        # 生成未来时间戳
        last_timestamp = df['timestamps'].iloc[-1]
        if len(df) > 1:
            time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
        else:
            time_diff = pd.Timedelta(minutes=15)
        
        y_timestamp = pd.date_range(
            start=last_timestamp + time_diff,
            periods=pred_len,
            freq=time_diff
        )
        
        # 确保 timestamps 是 Series
        if isinstance(x_timestamp, pd.DatetimeIndex):
            x_timestamp = pd.Series(x_timestamp, name='timestamps')
        if isinstance(y_timestamp, pd.DatetimeIndex):
            y_timestamp = pd.Series(y_timestamp, name='timestamps')
        
        # 执行预测
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count
        )
        
        # 分析趋势
        pred_close = pred_df['close'].values
        if len(pred_close) > 1:
            change_pct = (pred_close[-1] / pred_close[0] - 1) * 100
            if change_pct > 0.5:
                trend = 'up'
            elif change_pct < -0.5:
                trend = 'down'
            else:
                trend = 'sideways'
        else:
            change_pct = 0
            trend = 'sideways'
        
        return KronosPrediction(
            timestamps=y_timestamp.tolist(),
            open=pred_df['open'].values,
            high=pred_df['high'].values,
            low=pred_df['low'].values,
            close=pred_df['close'].values,
            volume=pred_df['volume'].values if 'volume' in pred_df.columns else None,
            trend=trend,
            change_pct=change_pct
        )
    
    def predict_batch(
        self,
        df_list: List[pd.DataFrame],
        pred_len: int = 6,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
    ) -> List[KronosPrediction]:
        """
        批量预测
        
        Args:
            df_list: K线数据列表
            pred_len: 预测长度
            T: 温度参数
            top_p: nucleus sampling 概率
            sample_count: 采样数量
            
        Returns:
            预测结果列表
        """
        # 准备批量输入
        df_list_processed = []
        xts_list = []
        yts_list = []
        
        for df in df_list:
            required_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                required_cols.append('volume')
            
            x_df = df[required_cols].tail(self.max_context)
            x_timestamp = df['timestamps'].tail(self.max_context)
            
            last_timestamp = df['timestamps'].iloc[-1]
            if len(df) > 1:
                time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
            else:
                time_diff = pd.Timedelta(minutes=15)
            
            y_timestamp = pd.date_range(
                start=last_timestamp + time_diff,
                periods=pred_len,
                freq=time_diff
            )
            
            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name='timestamps')
            if isinstance(y_timestamp, pd.DatetimeIndex):
                y_timestamp = pd.Series(y_timestamp, name='timestamps')
            
            df_list_processed.append(x_df)
            xts_list.append(x_timestamp)
            yts_list.append(y_timestamp)
        
        # 批量预测
        pred_dfs = self.predictor.predict_batch(
            df_list=df_list_processed,
            x_timestamp_list=xts_list,
            y_timestamp_list=yts_list,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False
        )
        
        # 处理结果
        results = []
        for pred_df in pred_dfs:
            pred_close = pred_df['close'].values
            if len(pred_close) > 1:
                change_pct = (pred_close[-1] / pred_close[0] - 1) * 100
                if change_pct > 0.5:
                    trend = 'up'
                elif change_pct < -0.5:
                    trend = 'down'
                else:
                    trend = 'sideways'
            else:
                change_pct = 0
                trend = 'sideways'
            
            results.append(KronosPrediction(
                timestamps=[],
                open=pred_df['open'].values,
                high=pred_df['high'].values,
                low=pred_df['low'].values,
                close=pred_df['close'].values,
                volume=pred_df['volume'].values if 'volume' in pred_df.columns else None,
                trend=trend,
                change_pct=change_pct
            ))
        
        return results


def create_predictor(
    model_name: str = "NeoQuasar/Kronos-base",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str = "cpu"
) -> Optional[KronosPredictorWrapper]:
    """
    创建 Kronos 预测器
    
    Args:
        model_name: 模型名称
        tokenizer_name: 分词器名称
        device: 设备
        
    Returns:
        预测器实例
    """
    if not MODEL_AVAILABLE:
        print("Kronos model not available")
        return None
    
    try:
        return KronosPredictorWrapper(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            device=device
        )
    except Exception as e:
        print(f"Failed to create predictor: {e}")
        return None


if __name__ == "__main__":
    # 测试
    print("Testing Kronos Predictor...")
    
    if not MODEL_AVAILABLE:
        print("Kronos model not available, skipping test")
    else:
        predictor = create_predictor()
        
        if predictor:
            # 创建测试数据
            dates = pd.date_range(start='2024-01-01', periods=500, freq='15min')
            np.random.seed(42)
            price = 40000 + np.cumsum(np.random.randn(500) * 100)
            
            df = pd.DataFrame({
                'timestamps': dates,
                'open': price + np.random.randn(500) * 50,
                'high': price + abs(np.random.randn(500) * 100),
                'low': price - abs(np.random.randn(500) * 100),
                'close': price,
                'volume': np.random.rand(500) * 1000
            })
            
            print("\nPredicting...")
            result = predictor.predict(df, pred_len=6)
            
            print(f"\nPrediction results:")
            print(f"Trend: {result.trend}")
            print(f"Change: {result.change_pct:.2f}%")
            print(f"Predicted close: {result.close}")
