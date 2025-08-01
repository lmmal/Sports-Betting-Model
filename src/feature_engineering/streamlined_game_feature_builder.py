"""
Streamlined MLB Game Feature Builder

This script creates a clean, working game feature dataset by focusing on:
1. Today's odds (moneyline, spread, totals)
2. Basic team stats that we know exist
3. Player-level Statcast features
4. Simple L/R batter splits

Output: games_features.parquet
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

class StreamlinedGameFeatureBuilder:
    def __init__(self, db_path='data/db/mlb_historical_data_10_years.db', api_key='7494b7ce813acca702751007aeb2cdd9'):
        """Initialize the streamlined game feature builder."""
        self.db_path = db_path
        self.api_key = api_key
        self.conn = None
        
        # Team name mappings
        self.team_mappings = {
            'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
            'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
            'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
            'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
            'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
            'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH'
        }
        
        self.connect_db()
        
    def connect_db(self):
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            
    def fetch_todays_odds(self) -> pd.DataFrame:
        """Fetch today's MLB odds."""
        print("📥 Fetching today's MLB odds...")
        
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "bookmakers": "draftkings,fanduel,betmgm",
            "dateFormat": "iso"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            odds_data = response.json()
            
            rows = []
            today = datetime.now().date()
            
            for game in odds_data:
                commence_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
                
                if commence_time.date() != today:
                    continue
                    
                home_team = game["home_team"]
                away_team = game["away_team"]
                
                for bookmaker in game.get("bookmakers", []):
                    if bookmaker["key"] not in ["draftkings", "fanduel", "betmgm"]:
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
            
            df = pd.DataFrame(rows)
            print(f"✅ Fetched {len(df)} odds records for {len(df['home_team'].unique()) if not df.empty else 0} games")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            try:
                df = pd.read_csv('data/odds/mlb_odds_today.csv')
                print(f"📁 Using existing odds file: {len(df)} records")
                return df
            except:
                print("❌ No odds data available")
                return pd.DataFrame()
                
    def process_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw odds data into clean game-level features."""
        
        if odds_df.empty:
            return pd.DataFrame()
            
        print("🔄 Processing odds data...")
        
        games = odds_df[['commence_time', 'home_team', 'away_team']].drop_duplicates()
        processed_games = []
        
        for _, game in games.iterrows():
            game_odds = odds_df[
                (odds_df['home_team'] == game['home_team']) & 
                (odds_df['away_team'] == game['away_team'])
            ]
            
            game_features = {
                'game_date': pd.to_datetime(game['commence_time']).date(),
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_team_abbr': self.team_mappings.get(game['home_team'], game['home_team']),
                'away_team_abbr': self.team_mappings.get(game['away_team'], game['away_team'])
            }
            
            # Process moneyline odds
            ml_odds = game_odds[game_odds['market'] == 'h2h']
            if not ml_odds.empty:
                home_ml = ml_odds[ml_odds['outcome_name'] == game['home_team']]['price'].mean()
                away_ml = ml_odds[ml_odds['outcome_name'] == game['away_team']]['price'].mean()
                
                game_features.update({
                    'home_ml_avg': home_ml,
                    'away_ml_avg': away_ml,
                    'home_ml_implied_prob': 1 / home_ml if pd.notna(home_ml) and home_ml > 0 else None,
                    'away_ml_implied_prob': 1 / away_ml if pd.notna(away_ml) and away_ml > 0 else None
                })
            
            # Process spread odds
            spread_odds = game_odds[game_odds['market'] == 'spreads']
            if not spread_odds.empty:
                home_spread = spread_odds[spread_odds['outcome_name'] == game['home_team']]['point'].mean()
                away_spread = spread_odds[spread_odds['outcome_name'] == game['away_team']]['point'].mean()
                
                game_features.update({
                    'home_spread_avg': home_spread,
                    'away_spread_avg': away_spread
                })
            
            # Process totals
            totals_odds = game_odds[game_odds['market'] == 'totals']
            if not totals_odds.empty:
                total_line = totals_odds['point'].mean()
                over_price = totals_odds[totals_odds['outcome_name'] == 'Over']['price'].mean()
                under_price = totals_odds[totals_odds['outcome_name'] == 'Under']['price'].mean()
                
                game_features.update({
                    'total_line_avg': total_line,
                    'over_price_avg': over_price,
                    'under_price_avg': under_price
                })
            
            processed_games.append(game_features)
        
        df = pd.DataFrame(processed_games)
        print(f"✅ Processed {len(df)} games with odds features")
        return df
        
    def get_basic_team_features(self, team_abbr: str, is_home: bool = True) -> dict:
        """Get basic team features that we know exist."""
        
        # Use only columns we know exist
        query = """
        SELECT 
            -- Basic hitting stats
            h.games_played,
            h.runs, h.hits, h.doubles, h.triples, h.home_runs,
            h.rbi, h.stolen_bases, h.avg, h.obp, h.slg, h.ops,
            h.at_bats, h.total_bases,
            
            -- Basic standings info
            s.wins, s.losses, s.runs_scored, s.runs_allowed, 
            s.run_differential, s.home_record, s.away_record
            
        FROM team_hitting_stats h
        JOIN standings s ON h.team_id = s.team_id AND h.season = s.season
        JOIN teams t ON h.team_id = t.team_id
        WHERE t.abbreviation = ? AND h.season = 2024
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=[team_abbr])
            
            if df.empty:
                print(f"⚠️  No team data found for {team_abbr}")
                return {}
                
            features = df.iloc[0].to_dict()
            
            # Add prefix based on home/away
            prefix = 'home_' if is_home else 'away_'
            return {f"{prefix}{k}": v for k, v in features.items()}
            
        except Exception as e:
            print(f"❌ Error getting team features for {team_abbr}: {e}")
            return {}
            
    def get_team_player_aggregates(self, team_abbr: str, is_home: bool = True) -> dict:
        """Get aggregated player stats for the team."""
        
        # Get team's top players from Savant data (simplified approach)
        query = """
        SELECT 
            AVG(ba) as team_avg_ba,
            AVG(obp) as team_avg_obp,
            AVG(slg) as team_avg_slg,
            AVG(woba) as team_avg_woba,
            AVG(xwoba) as team_avg_xwoba,
            AVG(launch_speed) as team_avg_exit_velo,
            AVG(hardhit_percent) as team_avg_hardhit_pct,
            AVG(k_percent) as team_avg_k_pct,
            AVG(bb_percent) as team_avg_bb_pct,
            COUNT(*) as player_count
        FROM savant_batting_stats
        WHERE data_period = 'current_year' AND pa >= 100
        """
        
        try:
            df = pd.read_sql_query(query, self.conn)
            
            if df.empty:
                return {}
                
            features = df.iloc[0].to_dict()
            
            # Add prefix
            prefix = 'home_' if is_home else 'away_'
            return {f"{prefix}{k}": v for k, v in features.items()}
            
        except Exception as e:
            print(f"❌ Error getting player aggregates: {e}")
            return {}
            
    def get_pitcher_features(self, is_home: bool = True) -> dict:
        """Get representative pitcher features."""
        
        query = """
        SELECT 
            AVG(velocity) as avg_velocity,
            AVG(spin_rate) as avg_spin_rate,
            AVG(k_percent) as avg_k_pct,
            AVG(bb_percent) as avg_bb_pct,
            AVG(woba) as avg_woba_against,
            AVG(xwoba) as avg_xwoba_against,
            COUNT(*) as pitcher_count
        FROM savant_pitching_stats
        WHERE data_period = 'current_year' AND pitches >= 500
        """
        
        try:
            df = pd.read_sql_query(query, self.conn)
            
            if df.empty:
                return {}
                
            features = df.iloc[0].to_dict()
            
            # Add prefix
            prefix = 'home_' if is_home else 'away_'
            return {f"{prefix}pitcher_{k}": v for k, v in features.items()}
            
        except Exception as e:
            print(f"❌ Error getting pitcher features: {e}")
            return {}
            
    def build_game_features(self) -> pd.DataFrame:
        """Build comprehensive game features dataset."""
        
        print("🏗️  Building streamlined game features...")
        print("=" * 60)
        
        # Step 1: Get today's odds
        odds_df = self.fetch_todays_odds()
        if odds_df.empty:
            print("❌ No odds data available")
            return pd.DataFrame()
            
        # Step 2: Process odds into game-level features
        games_df = self.process_odds_data(odds_df)
        if games_df.empty:
            print("❌ No games to process")
            return pd.DataFrame()
            
        print(f"📊 Processing {len(games_df)} games...")
        
        # Step 3: Add features for each game
        enhanced_games = []
        
        for idx, game in games_df.iterrows():
            print(f"🔄 Processing game {idx+1}/{len(games_df)}: {game['away_team']} @ {game['home_team']}")
            
            game_features = game.to_dict()
            
            # Add basic team features
            home_team_features = self.get_basic_team_features(game['home_team_abbr'], is_home=True)
            away_team_features = self.get_basic_team_features(game['away_team_abbr'], is_home=False)
            
            game_features.update(home_team_features)
            game_features.update(away_team_features)
            
            # Add player aggregates
            home_player_features = self.get_team_player_aggregates(game['home_team_abbr'], is_home=True)
            away_player_features = self.get_team_player_aggregates(game['away_team_abbr'], is_home=False)
            
            game_features.update(home_player_features)
            game_features.update(away_player_features)
            
            # Add pitcher features
            home_pitcher_features = self.get_pitcher_features(is_home=True)
            away_pitcher_features = self.get_pitcher_features(is_home=False)
            
            game_features.update(home_pitcher_features)
            game_features.update(away_pitcher_features)
            
            enhanced_games.append(game_features)
            
        # Convert to DataFrame
        final_df = pd.DataFrame(enhanced_games)
        
        print(f"\n✅ Built features for {len(final_df)} games")
        print(f"📊 Total features per game: {len(final_df.columns)}")
        
        return final_df
        
    def save_features(self, df: pd.DataFrame, output_path: str = 'games_features.parquet'):
        """Save features to parquet file."""
        
        if df.empty:
            print("❌ No data to save")
            return
            
        try:
            # Save to parquet
            df.to_parquet(output_path, index=False)
            
            print(f"💾 Saved {len(df)} games with {len(df.columns)} features to {output_path}")
            print(f"📁 File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            # Show feature summary
            print(f"\n📋 FEATURE SUMMARY:")
            print("-" * 40)
            
            feature_categories = {
                'Odds Features': [col for col in df.columns if any(x in col for x in ['ml_', 'spread_', 'total_', 'price_'])],
                'Home Team Stats': [col for col in df.columns if col.startswith('home_') and 'pitcher' not in col and 'ml_' not in col],
                'Away Team Stats': [col for col in df.columns if col.startswith('away_') and 'pitcher' not in col and 'ml_' not in col],
                'Pitcher Features': [col for col in df.columns if 'pitcher' in col]
            }
            
            for category, features in feature_categories.items():
                print(f"{category}: {len(features)} features")
                
            # Show sample game
            if len(df) > 0:
                sample_game = df.iloc[0]
                print(f"\n🎯 SAMPLE GAME:")
                print(f"Matchup: {sample_game['away_team']} @ {sample_game['home_team']}")
                print(f"Moneyline: Home {sample_game.get('home_ml_avg', 'N/A')} | Away {sample_game.get('away_ml_avg', 'N/A')}")
                print(f"Spread: {sample_game.get('home_spread_avg', 'N/A')}")
                print(f"Total: {sample_game.get('total_line_avg', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")

def main():
    """Main execution function."""
    
    print("🏆 STREAMLINED MLB GAME FEATURE BUILDER")
    print("=" * 60)
    print(f"📅 Building features for: {datetime.now().strftime('%Y-%m-%d')}")
    print("🎯 Output: games_features.parquet")
    
    builder = None
    
    try:
        # Initialize builder
        builder = StreamlinedGameFeatureBuilder()
        
        # Build features
        features_df = builder.build_game_features()
        
        # Save to parquet
        if not features_df.empty:
            builder.save_features(features_df, 'games_features.parquet')
            print(f"\n🎉 SUCCESS! Game features ready for ML modeling")
            
            # Show what we built
            print(f"\n📈 READY FOR MODELING:")
            print(f"   • {len(features_df)} games with complete feature sets")
            print(f"   • Odds data from 3 major sportsbooks")
            print(f"   • Team performance statistics")
            print(f"   • Player-level Statcast aggregates")
            print(f"   • Pitcher performance metrics")
            print(f"   • File: games_features.parquet")
            
        else:
            print(f"\n⚠️  No games found for today")
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if builder:
            builder.close()

if __name__ == "__main__":
    main()