import urllib.request
import json
import datetime

print("=== 1. PINGING BINANCE SPOT & FUTURES PUBLIC APIS ===")
# Binance Spot
try:
    url_spot = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=3'
    req = urllib.request.Request(url_spot, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        last_kline = data[-1]
        open_time = datetime.datetime.fromtimestamp(last_kline[0]/1000, tz=datetime.timezone.utc)
        close_time = datetime.datetime.fromtimestamp(last_kline[6]/1000, tz=datetime.timezone.utc)
        print(f"Binance Spot BTC/USDT:")
        print(f"  Latest 1m Bar Open:  {open_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Latest 1m Bar Close: {close_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Current Price:       ${float(last_kline[4]):,.2f}")
        print(f"  1m Volume:           {float(last_kline[5]):.4f} BTC")
except Exception as e:
    print("Binance Spot error:", e)

# Binance Futures
try:
    url_fut = 'https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=3'
    req = urllib.request.Request(url_fut, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        last_kline = data[-1]
        open_time = datetime.datetime.fromtimestamp(last_kline[0]/1000, tz=datetime.timezone.utc)
        print(f"Binance Futures BTCUSDT Perp:")
        print(f"  Latest 1m Bar Open:  {open_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Current Mark Price:  ${float(last_kline[4]):,.2f}")
except Exception as e:
    print("Binance Futures error:", e)

print("\n=== 2. PINGING HYPERLIQUID MAINNET API (Liquidity & Perpetuals) ===")
try:
    url_hl = 'https://api.hyperliquid.xyz/info'
    payload = json.dumps({'type': 'allMids'}).encode('utf-8')
    req = urllib.request.Request(url_hl, data=payload, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        mids = json.loads(resp.read().decode('utf-8'))
        print(f"Hyperliquid Status: LIVE CONNECTED (Total Tradable Perpetual Pairs: {len(mids)})")
        for coin in ['BTC', 'ETH', 'SOL', 'NVDA', 'COIN']:
            if coin in mids:
                print(f"  Hyperliquid {coin} Mid Price: ${float(mids[coin]):,.2f}")

    payload2 = json.dumps({'type': 'metaAndAssetCtxs'}).encode('utf-8')
    req2 = urllib.request.Request(url_hl, data=payload2, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req2, timeout=10) as resp:
        meta_ctx = json.loads(resp.read().decode('utf-8'))
        universe = meta_ctx[0]['universe']
        ctxs = meta_ctx[1]
        btc_idx = None
        for i, u in enumerate(universe):
            if u['name'] == 'BTC':
                btc_idx = i
                break
        if btc_idx is not None:
            btc_ctx = ctxs[btc_idx]
            oi_coins = float(btc_ctx['openInterest'])
            mark_px = float(btc_ctx['markPx'])
            oi_usd = oi_coins * mark_px
            funding_hr = float(btc_ctx['funding']) * 100.0
            day_vol = float(btc_ctx['dayNtlVlm'])
            print(f"Hyperliquid BTC Perpetual State:")
            print(f"  Mark Price:          ${mark_px:,.2f}")
            print(f"  Open Interest:       {oi_coins:,.2f} BTC (${oi_usd/1e6:,.2f}M USD)")
            print(f"  1-Hour Funding Rate: {funding_hr:.6f}% (Annualized: {funding_hr * 24 * 365:.2f}%)")
            print(f"  24h Notional Volume: ${day_vol/1e6:,.2f}M USD")
except Exception as e:
    print("Hyperliquid error:", e)

print("\n=== 3. PINGING ALTERNATIVE.ME FEAR & GREED API ===")
try:
    url_fg = 'https://api.alternative.me/fng/?limit=2'
    req = urllib.request.Request(url_fg, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        fg = json.loads(resp.read().decode('utf-8'))
        data = fg['data']
        for item in data:
            dt = datetime.datetime.fromtimestamp(int(item['timestamp']), tz=datetime.timezone.utc)
            print(f"  Date: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')} | Value: {item['value']} | Status: {item['value_classification']}")
except Exception as e:
    print("F&G error:", e)
