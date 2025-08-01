import requests
import csv
from datetime import datetime

API_KEY = "7494b7ce813acca702751007aeb2cdd9"
SPORT = "baseball_mlb"
REGION = "us"
BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]
MARKETS = ["h2h", "spreads", "totals"]

url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
params = {
    "apiKey": API_KEY,
    "regions": REGION,
    "markets": ",".join(MARKETS),
    "bookmakers": ",".join(BOOKMAKERS),
    "dateFormat": "iso"
}

response = requests.get(url, params=params)
response.raise_for_status()
odds_data = response.json()

today = datetime.now().date()
rows = []

for game in odds_data:
    commence_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
    if commence_time.date() != today:
        continue
    home_team = game["home_team"]
    away_team = game["away_team"]
    for bookmaker in game.get("bookmakers", []):
        if bookmaker["key"] not in BOOKMAKERS:
            continue
        for market in bookmaker.get("markets", []):
            market_type = market["key"]
            for outcome in market.get("outcomes", []):
                row = {
                    "commence_time": commence_time.isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bookmaker["key"],
                    "market": market_type,
                    "outcome_name": outcome.get("name"),
                    "price": outcome.get("price"),
                    "point": outcome.get("point")
                }
                rows.append(row)

with open("mlb_odds_today.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} odds rows to mlb_odds_today.csv")
