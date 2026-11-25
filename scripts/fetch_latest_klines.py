import urllib.request
import json
import datetime

url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=25"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as r:
    data = json.loads(r.read().decode())

print(f"Total 5m bars returned: {len(data)}")
for k in data:
    ot = datetime.datetime.fromtimestamp(k[0]/1000, tz=datetime.timezone.utc)
    ct = datetime.datetime.fromtimestamp(k[6]/1000, tz=datetime.timezone.utc)
    fmt_ot = ot.strftime("%Y-%m-%d %H:%M")
    fmt_ct = ct.strftime("%H:%M")
    print(f"{fmt_ot} -> {fmt_ct} | O: ${float(k[1]):.2f} H: ${float(k[2]):.2f} L: ${float(k[3]):.2f} C: ${float(k[4]):.2f} Vol: {float(k[5]):.2f} BTC TakerBuy: {float(k[9]):.2f} BTC")
