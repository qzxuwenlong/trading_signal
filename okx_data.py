"""
OKX 数据获取模块
支持获取加密货币、美股、大宗商品的K线数据
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
sys.path.insert(0, ".")
from config import OKX_BASE_URL, TRADING_PAIRS, TIMEFRAMES, PROXY_CONFIG


class OKXDataFetcher:
    """OKX 数据获取器"""
    
    def __init__(self, use_proxy: bool = None):
        self.base_url = OKX_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })
        
        # 代理配置
        if use_proxy is None:
            use_proxy = PROXY_CONFIG.get("enabled", True)
        
        if use_proxy:
            self.session.proxies = {
                'http': PROXY_CONFIG.get("http", "http://127.0.0.1:7890"),
                'https': PROXY_CONFIG.get("https", "http://127.0.0.1:7890"),
            }
            self.session.verify = PROXY_CONFIG.get("verify_ssl", False)
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def get_klines(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 300,
        after: str = "",
        before: str = ""
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对 (如 "BTC-USDT")
            timeframe: K线周期 (1m, 5m, 15m, 1H, 4H, 1D)
            limit: 返回数量 (最大300)
            after: 分页参数，返回此时间戳之后的数据
            before: 分页参数，返回此时间戳之前的数据
            
        Returns:
            DataFrame with columns: [timestamps, open, high, low, close, volume, amount]
        """
        endpoint = f"{self.base_url}/api/v5/market/candles"
        
        params = {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit),
        }
        
        if after:
            params["after"] = after
        if before:
            params["before"] = before
            
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "0":
                print(f"API Error: {data.get('msg', 'Unknown error')}")
                return pd.DataFrame()
            
            klines = data.get("data", [])
            if not klines:
                return pd.DataFrame()
            
            # OKX 返回格式: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            records = []
            for k in klines:
                records.append({
                    "timestamps": pd.to_datetime(int(k[0]), unit='ms'),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "amount": float(k[6]),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamps").reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    def get_klines_full(
        self,
        symbol: str,
        timeframe: str = "15m",
        total_bars: int = 600
    ) -> pd.DataFrame:
        """
        获取完整K线数据（自动分页）
        
        Args:
            symbol: 交易对
            timeframe: K线周期
            total_bars: 总共需要的K线数量
            
        Returns:
            完整的K线DataFrame
        """
        all_klines = []
        after = ""
        
        while len(all_klines) < total_bars:
            if after:
                df = self.get_klines(symbol, timeframe, limit=300, after=after)
            else:
                df = self.get_klines(symbol, timeframe, limit=300)
            
            if df.empty:
                break
                
            all_klines.append(df)
            
            # 获取最后一条的时间戳作为分页参数
            after = str(int(df["timestamps"].iloc[-1].timestamp() * 1000))
            
            # 如果返回数量少于300，说明没有更多数据
            if len(df) < 300:
                break
            
            # 避免请求过快
            time.sleep(0.1)
        
        if not all_klines:
            return pd.DataFrame()
        
        result = pd.concat(all_klines, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamps"], keep="last")
        result = result.sort_values("timestamps").reset_index(drop=True)
        
        # 只取最后 total_bars 条
        if len(result) > total_bars:
            result = result.tail(total_bars).reset_index(drop=True)
        
        return result
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        获取最新行情
        
        Args:
            symbol: 交易对
            
        Returns:
            最新行情数据
        """
        endpoint = f"{self.base_url}/api/v5/market/ticker"
        
        params = {"instId": symbol}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "0":
                return {}
            
            ticker = data.get("data", [{}])[0]
            
            return {
                "symbol": symbol,
                "last": float(ticker.get("last", 0)),
                "bid": float(ticker.get("bidPx", 0)),
                "ask": float(ticker.get("askPx", 0)),
                "high24h": float(ticker.get("high24h", 0)),
                "low24h": float(ticker.get("low24h", 0)),
                "vol24h": float(ticker.get("vol24h", 0)),
                "change24h": float(ticker.get("sodUtc8", 0)),
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """获取所有可用的交易对"""
        endpoint = f"{self.base_url}/api/v5/market/instruments"
        
        try:
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "0":
                return []
            
            instruments = data.get("data", [])
            symbols = [inst["instId"] for inst in instruments if inst.get("instType") == "SPOT"]
            
            return symbols
            
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []


def get_market_data(
    symbol: str = "BTC-USDT",
    timeframe: str = "15m",
    total_bars: int = 600
) -> pd.DataFrame:
    """
    便捷函数：获取市场数据
    
    Args:
        symbol: 交易对
        timeframe: K线周期
        total_bars: 总共需要的K线数量
        
    Returns:
        K线DataFrame
    """
    fetcher = OKXDataFetcher()
    return fetcher.get_klines_full(symbol, timeframe, total_bars)


if __name__ == "__main__":
    # 测试
    fetcher = OKXDataFetcher()
    
    print("获取 BTC-USDT 15m K线数据...")
    df = fetcher.get_klines_full("BTC-USDT", "15m", 600)
    print(f"获取 {len(df)} 条数据")
    print(f"时间范围: {df['timestamps'].min()} ~ {df['timestamps'].max()}")
    print(f"价格范围: ${df['close'].min():.2f} ~ ${df['close'].max():.2f}")
    print("\n最新5条数据:")
    print(df.tail())
