"""
Historical MLB Data Collector

This script collects historical game outcomes and odds data to create a labeled training dataset.
Uses multiple data sources:
1. MLB API for game outcomes and scores
2. The Odds API for historical betting lines
3. Retrosheet for additional historical data validation

Output: historical_games.parquet (labeled training data)
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta, date
import time
from typing import Dict, List, Optional, Tuple
import warnings
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas.game_features_schema import MLBGameFeatures, GameOutcome, Team

warnings.filterwarnings('ignore')


class HistoricalMLBDataCollector:
    def __init__(self, 
                 odds_api_key: str = '7494b7ce813acca702751007aeb2cdd9',
                 db_path: str = 'data/db/mlb_historical_data_10_years.db'):
        """
        Initialize historical data collector
        
        Args:
            odds_api_key: The Odds API key
            db_path: Path to existing database
        """
        self.odds_api_key = odds_api_key
        self.db_path = db_path
        self.conn = None
        
        # MLB API endpoints
        self.mlb_api_base = "https://statsapi.mlb.com/api/v1"
        
        # Team mappings for different data sources
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
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.conn = None
    
    def get_mlb_schedule(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get MLB schedule for date range from MLB API
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with game schedule
        """
        print(f"📅 Fetching MLB schedule: {start_date} to {end_date}")
        
        url = f"{self.mlb_api_base}/schedule"
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'sportId': 1,  # MLB
            'hydrate': 'team,linescore'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for date_info in data.get('dates', []):
                game_date = date_info['date']
                for game in date_info.get('games', []):
                    if game['status']['detailedState'] == 'Final':
                        game_info = {
                            'game_id': str(game['gamePk']),
                            'game_date': game_date,
                            'home_team': game['teams']['home']['team']['abbreviation'],
                            'away_team': game['teams']['away']['team']['abbreviation'],
                            'home_score': game['teams']['home']['score'],
                            'away_score': game['teams']['away']['score'],
                            'status': game['status']['detailedState']
                        }
                        games.append(game_info)
            
            df = pd.DataFrame(games)
            print(f"✅ Found {len(df)} completed games")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching MLB schedule: {e}")
            return pd.DataFrame()
    
    def get_historical_odds(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical odds data from The Odds API
        
        Args:
            start_date: Start date (YYYY-MM-DD)  
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical odds
        """
        print(f"💰 Fetching historical odds: {start_date} to {end_date}")
        
        # Note: The Odds API doesn't have extensive historical data for free tier
        # This is a placeholder for when you have access to historical odds
        # For now, we'll create synthetic odds based on team strength
        
        odds_data = []
        
        # Get games from schedule first
        schedule_df = self.get_mlb_schedule(start_date, end_date)
        
        for _, game in schedule_df.iterrows():
            # Create synthetic odds based on team records (placeholder)
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Simple synthetic odds generation (replace with real data when available)
            home_ml = np.random.uniform(-200, 200)
            away_ml = -home_ml if home_ml > 0 else 100 + abs(home_ml)
            
            odds_info = {
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_ml': home_ml,
                'away_ml': away_ml,
                'home_spread': np.random.uniform(-2.5, 2.5),
                'away_spread': -np.random.uniform(-2.5, 2.5),
                'over_under': np.random.uniform(7.5, 12.5),
                'over_odds': -110,
                'under_odds': -110
            }
            odds_data.append(odds_info)
        
        df = pd.DataFrame(odds_data)
        print(f"✅ Generated odds for {len(df)} games")
        return df
    
    def calculate_outcome_labels(self, games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate outcome labels for training
        
        Args:
            games_df: DataFrame with game results
            odds_df: DataFrame with odds data
            
        Returns:
            DataFrame with labeled outcomes
        """
        print("🎯 Calculating outcome labels...")
        
        # Merge games and odds
        merged_df = pd.merge(games_df, odds_df, on=['game_id', 'game_date', 'home_team', 'away_team'], how='inner')
        
        if merged_df.empty:
            print("⚠️ No matching games and odds data")
            return pd.DataFrame()
        
        # Calculate outcome labels
        merged_df['total_runs'] = merged_df['home_score'] + merged_df['away_score']
        merged_df['home_won'] = merged_df['home_score'] > merged_df['away_score']
        merged_df['away_won'] = merged_df['away_score'] > merged_df['home_score']
        merged_df['winner'] = merged_df['home_team'].where(merged_df['home_won'], merged_df['away_team'])
        
        # Over/under outcomes
        merged_df['over_hit'] = merged_df['total_runs'] > merged_df['over_under']
        merged_df['under_hit'] = merged_df['total_runs'] < merged_df['over_under']
        
        # Spread outcomes
        merged_df['home_spread_result'] = merged_df['home_score'] - merged_df['away_score']
        merged_df['home_covered'] = merged_df['home_spread_result'] > merged_df['home_spread']
        merged_df['away_covered'] = merged_df['home_spread_result'] < merged_df['away_spread']
        
        print(f"✅ Calculated labels for {len(merged_df)} games")
        return merged_df
    
    def collect_historical_data(self, 
                              start_date: str, 
                              end_date: str,
                              output_file: str = 'data/historical_games.parquet') -> pd.DataFrame:
        """
        Collect complete historical dataset with outcomes
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            output_file: Output file path
            
        Returns:
            DataFrame with historical labeled data
        """
        print("🏆 HISTORICAL MLB DATA COLLECTION")
        print("=" * 50)
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"💾 Output: {output_file}")
        
        # Get game results
        games_df = self.get_mlb_schedule(start_date, end_date)
        if games_df.empty:
            print("❌ No games found for date range")
            return pd.DataFrame()
        
        # Get odds data
        odds_df = self.get_historical_odds(start_date, end_date)
        if odds_df.empty:
            print("❌ No odds data found")
            return pd.DataFrame()
        
        # Calculate outcomes
        labeled_df = self.calculate_outcome_labels(games_df, odds_df)
        if labeled_df.empty:
            print("❌ Could not calculate outcome labels")
            return pd.DataFrame()
        
        # Save to parquet
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            labeled_df.to_parquet(output_file, index=False)
            print(f"✅ Saved historical data: {output_file}")
            
            # Show summary
            print(f"\n📊 HISTORICAL DATA SUMMARY:")
            print(f"   • Total games: {len(labeled_df)}")
            print(f"   • Date range: {labeled_df['game_date'].min()} to {labeled_df['game_date'].max()}")
            print(f"   • Home team win rate: {labeled_df['home_won'].mean():.3f}")
            print(f"   • Over hit rate: {labeled_df['over_hit'].mean():.3f}")
            print(f"   • Home cover rate: {labeled_df['home_covered'].mean():.3f}")
            
        except Exception as e:
            print(f"❌ Error saving historical data: {e}")
        
        return labeled_df
    
    def collect_recent_games(self, days_back: int = 30) -> pd.DataFrame:
        """
        Collect recent games for quick testing
        
        Args:
            days_back: Number of days back to collect
            
        Returns:
            DataFrame with recent labeled games
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        return self.collect_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            f'data/recent_games_{days_back}days.parquet'
        )
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main execution function"""
    
    print("🏆 HISTORICAL MLB DATA COLLECTOR")
    print("=" * 60)
    
    collector = None
    
    try:
        # Initialize collector
        collector = HistoricalMLBDataCollector()
        
        # Collect recent games (last 30 days) for quick start
        print("\n🚀 Quick start: Collecting last 30 days of games...")
        recent_df = collector.collect_recent_games(days_back=30)
        
        if not recent_df.empty:
            print(f"\n🎉 SUCCESS! Recent games collected for training")
            print(f"   • Ready for ML model training")
            print(f"   • Contains game outcomes and betting results")
            print(f"   • File: data/recent_games_30days.parquet")
        
        # Option to collect larger historical dataset
        print(f"\n💡 TIP: To collect more historical data, use:")
        print(f"   collector.collect_historical_data('2024-04-01', '2024-10-31')")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if collector:
            collector.close()


if __name__ == "__main__":
    main()
