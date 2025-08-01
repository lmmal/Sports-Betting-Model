"""
Comprehensive MLB Database Analyzer

This script provides advanced analysis tools for the complete MLB database,
including team-level historical data and player-level Statcast data from Baseball Savant.

Features:
- Team performance analysis over time
- Player performance analysis with Statcast metrics
- Cross-reference team and player data
- Advanced statistical analysis and correlations
- Export capabilities for further analysis
- Betting insights and value identification

Author: Sports Betting Model
Date: 2024
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class ComprehensiveMLBAnalyzer:
    def __init__(self, db_path='mlb_historical_data_10_years.db'):
        """
        Initialize the Comprehensive MLB Database Analyzer.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.connect_db()
        
    def connect_db(self):
        """Create database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            
    def get_database_overview(self):
        """Get comprehensive overview of all data in the database."""
        cursor = self.conn.cursor()
        
        print("📊 DATABASE OVERVIEW")
        print("=" * 50)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"📋 Total Tables: {len(tables)}")
        print("\n🗂️  TABLE DETAILS:")
        
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
                
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get date range if applicable
            date_info = ""
            if 'season' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]:
                cursor.execute(f"SELECT MIN(season), MAX(season) FROM {table_name}")
                min_year, max_year = cursor.fetchone()
                if min_year and max_year:
                    date_info = f" ({min_year}-{max_year})"
            
            print(f"   📊 {table_name}: {count:,} records{date_info}")
            
    def get_top_players_by_metric(self, metric: str, data_period: str = '10_years', 
                                 player_type: str = 'batting', top_n: int = 20):
        """
        Get top players by a specific Statcast metric.
        
        Args:
            metric (str): Statcast metric to rank by
            data_period (str): Data period to analyze
            player_type (str): 'batting' or 'pitching'
            top_n (int): Number of top players to return
        """
        
        table_name = f'savant_{player_type}_stats'
        
        # Determine sort order based on metric
        ascending_metrics = ['era', 'whip', 'bb_percent', 'xba', 'babip']
        sort_order = 'ASC' if any(m in metric.lower() for m in ascending_metrics) else 'DESC'
        
        query = f"""
        SELECT 
            player_name,
            player_id,
            {metric},
            pa,
            CASE 
                WHEN {metric} IS NOT NULL THEN 
                    RANK() OVER (ORDER BY {metric} {sort_order})
                ELSE NULL 
            END as rank
        FROM {table_name}
        WHERE data_period = ? 
            AND {metric} IS NOT NULL
            AND pa >= 100
        ORDER BY {metric} {sort_order}
        LIMIT ?
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=[data_period, top_n])
            return df
        except Exception as e:
            print(f"❌ Error getting top players for {metric}: {e}")
            return pd.DataFrame()
            
    def compare_players(self, player_names: List[str], data_period: str = '10_years'):
        """Compare multiple players across key metrics."""
        
        placeholders = ','.join(['?' for _ in player_names])
        
        # Batting comparison
        batting_query = f"""
        SELECT 
            player_name,
            player_id,
            data_period,
            pa,
            ba,
            obp,
            slg,
            woba,
            xwoba,
            launch_speed,
            launch_angle,
            hardhit_percent,
            barrel_batted_rate,
            k_percent,
            bb_percent
        FROM savant_batting_stats
        WHERE player_name IN ({placeholders}) 
            AND data_period = ?
        """
        
        # Pitching comparison  
        pitching_query = f"""
        SELECT 
            player_name,
            player_id,
            data_period,
            pa,
            ba,
            woba,
            xwoba,
            velocity,
            spin_rate,
            k_percent,
            bb_percent,
            hardhit_percent,
            barrel_batted_rate
        FROM savant_pitching_stats
        WHERE player_name IN ({placeholders}) 
            AND data_period = ?
        """
        
        params = player_names + [data_period]
        
        try:
            batting_df = pd.read_sql_query(batting_query, self.conn, params=params)
            pitching_df = pd.read_sql_query(pitching_query, self.conn, params=params)
            
            return {
                'batting': batting_df,
                'pitching': pitching_df
            }
        except Exception as e:
            print(f"❌ Error comparing players: {e}")
            return {'batting': pd.DataFrame(), 'pitching': pd.DataFrame()}
            
    def get_team_player_correlation(self, team_abbr: str, season: int):
        """
        Analyze correlation between team performance and individual player metrics.
        """
        
        # Get team stats for the season
        team_query = """
        SELECT 
            team_id,
            season,
            games_played,
            runs,
            hits,
            home_runs,
            avg,
            obp,
            slg,
            ops
        FROM team_hitting_stats
        WHERE season = ?
        """
        
        # Get players who played for the team (this is approximate since we don't have 
        # exact team-player mappings by season in the Savant data)
        player_query = """
        SELECT 
            player_name,
            player_id,
            ba,
            obp,
            slg,
            woba,
            xwoba,
            launch_speed,
            hardhit_percent,
            k_percent,
            bb_percent
        FROM savant_batting_stats
        WHERE data_period = 'current_year'
        ORDER BY woba DESC
        LIMIT 50
        """
        
        try:
            team_df = pd.read_sql_query(team_query, self.conn, params=[season])
            player_df = pd.read_sql_query(player_query, self.conn)
            
            return {
                'team_stats': team_df,
                'top_players': player_df
            }
        except Exception as e:
            print(f"❌ Error analyzing team-player correlation: {e}")
            return {'team_stats': pd.DataFrame(), 'top_players': pd.DataFrame()}
            
    def identify_breakout_candidates(self, data_period: str = '3_years'):
        """
        Identify potential breakout candidates based on underlying metrics.
        """
        
        query = """
        SELECT 
            player_name,
            player_id,
            ba,
            xba,
            (xba - ba) as ba_upside,
            obp,
            xobp,
            (xobp - obp) as obp_upside,
            slg,
            xslg,
            (xslg - slg) as slg_upside,
            woba,
            xwoba,
            (xwoba - woba) as woba_upside,
            launch_speed,
            launch_angle,
            hardhit_percent,
            barrel_batted_rate,
            pa
        FROM savant_batting_stats
        WHERE data_period = ?
            AND pa >= 200
            AND xba > ba
            AND xwoba > woba
        ORDER BY (xwoba - woba) DESC
        LIMIT 25
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=[data_period])
            return df
        except Exception as e:
            print(f"❌ Error identifying breakout candidates: {e}")
            return pd.DataFrame()
            
    def identify_regression_candidates(self, data_period: str = '3_years'):
        """
        Identify players who may be due for regression based on underlying metrics.
        """
        
        query = """
        SELECT 
            player_name,
            player_id,
            ba,
            xba,
            (ba - xba) as ba_overperformance,
            obp,
            xobp,
            (obp - xobp) as obp_overperformance,
            slg,
            xslg,
            (slg - xslg) as slg_overperformance,
            woba,
            xwoba,
            (woba - xwoba) as woba_overperformance,
            launch_speed,
            launch_angle,
            hardhit_percent,
            barrel_batted_rate,
            babip,
            pa
        FROM savant_batting_stats
        WHERE data_period = ?
            AND pa >= 200
            AND ba > xba
            AND woba > xwoba
            AND babip > 0.320
        ORDER BY (woba - xwoba) DESC
        LIMIT 25
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=[data_period])
            return df
        except Exception as e:
            print(f"❌ Error identifying regression candidates: {e}")
            return pd.DataFrame()
            
    def get_pitcher_stuff_rankings(self, data_period: str = '10_years', min_pitches: int = 1000):
        """
        Rank pitchers by 'stuff' metrics like velocity and spin rate.
        """
        
        query = """
        SELECT 
            player_name,
            player_id,
            velocity,
            spin_rate,
            k_percent,
            bb_percent,
            (k_percent - bb_percent) as k_minus_bb,
            whiffs,
            swings,
            CASE WHEN swings > 0 THEN (whiffs * 100.0 / swings) ELSE 0 END as whiff_rate,
            hardhit_percent,
            barrel_batted_rate,
            xwoba,
            pitches
        FROM savant_pitching_stats
        WHERE data_period = ?
            AND pitches >= ?
        ORDER BY velocity DESC, spin_rate DESC
        LIMIT 50
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=[data_period, min_pitches])
            return df
        except Exception as e:
            print(f"❌ Error getting pitcher stuff rankings: {e}")
            return pd.DataFrame()
            
    def export_analysis_data(self, analysis_type: str, output_file: str, **kwargs):
        """
        Export analysis results to CSV for further analysis.
        """
        
        analysis_functions = {
            'breakout_candidates': self.identify_breakout_candidates,
            'regression_candidates': self.identify_regression_candidates,
            'pitcher_stuff': self.get_pitcher_stuff_rankings,
            'top_players': self.get_top_players_by_metric
        }
        
        if analysis_type not in analysis_functions:
            print(f"❌ Unknown analysis type: {analysis_type}")
            return
            
        try:
            df = analysis_functions[analysis_type](**kwargs)
            
            if not df.empty:
                df.to_csv(output_file, index=False)
                print(f"✅ Exported {len(df)} records to {output_file}")
            else:
                print("⚠️  No data to export")
                
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            
    def generate_comprehensive_report(self, output_file: str = None):
        """
        Generate a comprehensive analysis report.
        """
        
        report = []
        report.append("🏆 COMPREHENSIVE MLB DATABASE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Database overview
        report.append("📊 DATABASE OVERVIEW")
        report.append("-" * 30)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
                
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            report.append(f"{table_name}: {count:,} records")
            
        # Top performers analysis
        report.append("\n🏏 TOP BATTING PERFORMERS (10-Year Period)")
        report.append("-" * 40)
        
        top_woba = self.get_top_players_by_metric('woba', '10_years', 'batting', 10)
        if not top_woba.empty:
            for _, player in top_woba.iterrows():
                report.append(f"{player['rank']:2d}. {player['player_name']}: {player['woba']:.3f} wOBA")
                
        report.append("\n⚾ TOP PITCHING PERFORMERS (10-Year Period)")
        report.append("-" * 40)
        
        top_k_rate = self.get_top_players_by_metric('k_percent', '10_years', 'pitching', 10)
        if not top_k_rate.empty:
            for _, player in top_k_rate.iterrows():
                report.append(f"{player['rank']:2d}. {player['player_name']}: {player['k_percent']:.1f}% K Rate")
                
        # Breakout candidates
        report.append("\n🚀 BREAKOUT CANDIDATES")
        report.append("-" * 25)
        
        breakouts = self.identify_breakout_candidates('3_years')
        if not breakouts.empty:
            for _, player in breakouts.head(5).iterrows():
                report.append(f"{player['player_name']}: +{player['woba_upside']:.3f} wOBA upside")
                
        # Regression candidates
        report.append("\n📉 REGRESSION CANDIDATES")
        report.append("-" * 25)
        
        regressions = self.identify_regression_candidates('3_years')
        if not regressions.empty:
            for _, player in regressions.head(5).iterrows():
                report.append(f"{player['player_name']}: -{player['woba_overperformance']:.3f} wOBA overperformance")
                
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"✅ Report saved to {output_file}")
        else:
            print(report_text)
            
        return report_text
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")

def main():
    """Main execution function with interactive menu."""
    
    analyzer = ComprehensiveMLBAnalyzer()
    
    try:
        while True:
            print("\n🏆 COMPREHENSIVE MLB ANALYZER")
            print("=" * 40)
            print("1. Database Overview")
            print("2. Top Players by Metric")
            print("3. Compare Players")
            print("4. Breakout Candidates")
            print("5. Regression Candidates") 
            print("6. Pitcher Stuff Rankings")
            print("7. Generate Full Report")
            print("8. Export Analysis Data")
            print("9. Exit")
            
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                analyzer.get_database_overview()
                
            elif choice == '2':
                metric = input("Enter metric (e.g., woba, k_percent, velocity): ").strip()
                period = input("Enter period (10_years, 5_years, 3_years, current_year, last_year): ").strip() or '10_years'
                player_type = input("Enter type (batting, pitching): ").strip() or 'batting'
                
                df = analyzer.get_top_players_by_metric(metric, period, player_type)
                if not df.empty:
                    print(f"\n🏆 Top {len(df)} Players by {metric}:")
                    print(df.to_string(index=False))
                    
            elif choice == '3':
                players = input("Enter player names (comma-separated): ").strip().split(',')
                players = [p.strip() for p in players]
                period = input("Enter period: ").strip() or '10_years'
                
                comparison = analyzer.compare_players(players, period)
                if not comparison['batting'].empty:
                    print("\n🏏 Batting Comparison:")
                    print(comparison['batting'].to_string(index=False))
                if not comparison['pitching'].empty:
                    print("\n⚾ Pitching Comparison:")
                    print(comparison['pitching'].to_string(index=False))
                    
            elif choice == '4':
                period = input("Enter period: ").strip() or '3_years'
                df = analyzer.identify_breakout_candidates(period)
                if not df.empty:
                    print(f"\n🚀 Breakout Candidates ({period}):")
                    print(df[['player_name', 'ba', 'xba', 'ba_upside', 'woba', 'xwoba', 'woba_upside']].to_string(index=False))
                    
            elif choice == '5':
                period = input("Enter period: ").strip() or '3_years'
                df = analyzer.identify_regression_candidates(period)
                if not df.empty:
                    print(f"\n📉 Regression Candidates ({period}):")
                    print(df[['player_name', 'ba', 'xba', 'ba_overperformance', 'woba', 'xwoba', 'woba_overperformance']].to_string(index=False))
                    
            elif choice == '6':
                period = input("Enter period: ").strip() or '10_years'
                df = analyzer.get_pitcher_stuff_rankings(period)
                if not df.empty:
                    print(f"\n⚾ Pitcher Stuff Rankings ({period}):")
                    print(df[['player_name', 'velocity', 'spin_rate', 'k_percent', 'whiff_rate']].head(20).to_string(index=False))
                    
            elif choice == '7':
                output_file = input("Enter output file (optional): ").strip() or None
                analyzer.generate_comprehensive_report(output_file)
                
            elif choice == '8':
                analysis_type = input("Enter analysis type (breakout_candidates, regression_candidates, pitcher_stuff, top_players): ").strip()
                output_file = input("Enter output CSV file: ").strip()
                
                if analysis_type == 'top_players':
                    metric = input("Enter metric: ").strip()
                    period = input("Enter period: ").strip() or '10_years'
                    player_type = input("Enter type (batting, pitching): ").strip() or 'batting'
                    analyzer.export_analysis_data(analysis_type, output_file, 
                                                metric=metric, data_period=period, player_type=player_type)
                else:
                    period = input("Enter period: ").strip() or '3_years'
                    analyzer.export_analysis_data(analysis_type, output_file, data_period=period)
                    
            elif choice == '9':
                break
                
            else:
                print("❌ Invalid option. Please try again.")
                
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()