"""
主程序 - 交易信号生成器
支持命令行和 Web UI
"""
import sys
import os
import json
import argparse
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from okx_data import OKXDataFetcher, get_market_data
from indicators import TechnicalIndicators, analyze_technicals
from advanced_indicators import AdvancedIndicators, analyze_all_indicators
from kronos_model import KronosPredictorWrapper, create_predictor
from signal_generator import SignalGenerator, TradingSignal, format_signal
from enhanced_signal_generator import EnhancedSignalGenerator, TradingSignal as EnhancedTradingSignal, format_signal as enhanced_format_signal
from config import TRADING_PAIRS, TIMEFRAMES, WEB_CONFIG


app = Flask(__name__)
CORS(app)

# 全局变量
predictor = None
generator = None
enhanced_generator = None


def init_kronos():
    """初始化 Kronos 模型"""
    global predictor, generator, enhanced_generator
    
    print("Initializing Kronos model...")
    try:
        predictor = create_predictor()
        generator = SignalGenerator(kronos_predictor=predictor)
        enhanced_generator = EnhancedSignalGenerator(kronos_predictor=predictor)
        print("Kronos model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize Kronos: {e}")
        generator = SignalGenerator(kronos_predictor=None)
        enhanced_generator = EnhancedSignalGenerator(kronos_predictor=None)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/signal', methods=['POST'])
def get_signal():
    """获取交易信号"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USDT')
        timeframe = data.get('timeframe', '15m')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 6))
        use_enhanced = data.get('use_enhanced', True)
        
        if use_enhanced:
            if enhanced_generator is None:
                return jsonify({'error': 'Enhanced signal generator not initialized'}), 500
            signal = enhanced_generator.generate_signal(symbol, timeframe, lookback, pred_len)
        else:
            if generator is None:
                return jsonify({'error': 'Signal generator not initialized'}), 500
            signal = generator.generate_signal(symbol, timeframe, lookback, pred_len)
        
        return jsonify({
            'success': True,
            'signal': {
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'reasons': signal.reasons,
                'timestamp': signal.timestamp,
                'kronos_prediction': {
                    'trend': signal.kronos_prediction.trend if signal.kronos_prediction else None,
                    'change_pct': signal.kronos_prediction.change_pct if signal.kronos_prediction else 0,
                    'close': signal.kronos_prediction.close.tolist() if signal.kronos_prediction else [],
                } if signal.kronos_prediction else None,
                'indicators': signal.indicators if use_enhanced else None,
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals/batch', methods=['POST'])
def get_batch_signals():
    """批量获取交易信号"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTC-USDT', 'ETH-USDT'])
        timeframe = data.get('timeframe', '15m')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 6))
        
        if generator is None:
            return jsonify({'error': 'Signal generator not initialized'}), 500
        
        signals = generator.generate_signals_for_multiple(symbols, timeframe, lookback, pred_len)
        
        results = []
        for signal in signals:
            results.append({
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'reasons': signal.reasons,
                'timestamp': signal.timestamp,
            })
        
        return jsonify({
            'success': True,
            'signals': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/indicators', methods=['POST'])
def get_indicators():
    """获取技术指标"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USDT')
        timeframe = data.get('timeframe', '15m')
        
        df = get_market_data(symbol, timeframe, 200)
        if df.empty:
            return jsonify({'error': 'Failed to fetch data'}), 500
        
        indicators = analyze_technicals(df)
        
        return jsonify({
            'success': True,
            'indicators': indicators
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/klines', methods=['POST'])
def get_klines():
    """获取K线数据"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USDT')
        timeframe = data.get('timeframe', '15m')
        limit = int(data.get('limit', 100))
        
        fetcher = OKXDataFetcher()
        df = fetcher.get_klines_full(symbol, timeframe, limit)
        
        if df.empty:
            return jsonify({'error': 'Failed to fetch data'}), 500
        
        return jsonify({
            'success': True,
            'klines': df.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ticker', methods=['POST'])
def get_ticker():
    """获取最新行情"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USDT')
        
        fetcher = OKXDataFetcher()
        ticker = fetcher.get_ticker(symbol)
        
        return jsonify({
            'success': True,
            'ticker': ticker
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config')
def get_config():
    """获取配置"""
    return jsonify({
        'trading_pairs': TRADING_PAIRS,
        'timeframes': TIMEFRAMES,
        'kronos_available': predictor is not None,
    })


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='Kronos Trading Signal Generator')
    parser.add_argument('--symbol', default='BTC-USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--lookback', type=int, default=400, help='Lookback length')
    parser.add_argument('--pred-len', type=int, default=6, help='Prediction length')
    parser.add_argument('--web', action='store_true', help='Start Web UI')
    parser.add_argument('--no-kronos', action='store_true', help='Disable Kronos model')
    parser.add_argument('--enhanced', action='store_true', default=True, help='Use enhanced signal generator')
    
    args = parser.parse_args()
    
    if args.web:
        # 启动 Web UI
        print("Starting Web UI...")
        if not args.no_kronos:
            init_kronos()
        else:
            enhanced_generator = EnhancedSignalGenerator(kronos_predictor=None)
        app.run(
            host=WEB_CONFIG['host'],
            port=WEB_CONFIG['port'],
            debug=WEB_CONFIG['debug']
        )
    else:
        # 命令行模式
        if not args.no_kronos:
            init_kronos()
        
        if args.enhanced:
            gen = EnhancedSignalGenerator(kronos_predictor=predictor)
            signal = gen.generate_signal(args.symbol, args.timeframe, args.lookback, args.pred_len)
            print(enhanced_format_signal(signal))
        else:
            gen = SignalGenerator(kronos_predictor=predictor)
            signal = gen.generate_signal(args.symbol, args.timeframe, args.lookback, args.pred_len)
            print(format_signal(signal))


if __name__ == "__main__":
    main()
