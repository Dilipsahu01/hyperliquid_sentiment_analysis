import urllib.request
import json
import datetime

url = "https://api.hyperliquid.xyz/info"
payload = json.dumps({
    "type": "candleSnapshot",
    "req": {
        "coin": "BTC",
        "interval": "5m",
        "startTime": int(datetime.datetime(2026, 11, 25, 11, 45, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    }
}).encode()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        hl_candles = json.loads(r.read().decode())
        print(f"Hyperliquid 5m candles returned: {len(hl_candles)}")
        for c in hl_candles[-15:]:
            ot = datetime.datetime.fromtimestamp(c["t"]/1000, tz=datetime.timezone.utc)
            fmt_ot = ot.strftime("%H:%M")
            print(f"HL {fmt_ot} | O: ${float(c['o']):.2f} H: ${float(c['h']):.2f} L: ${float(c['l']):.2f} C: ${float(c['c']):.2f} V: {float(c['v']):.2f} BTC Trades: {c['n']}")
except Exception as e:
    print("Hyperliquid candle error:", e)
