"""
Fixed MLB Database Analyzer with correct column names

Quick analyzer to test the Baseball Savant data with proper column references.
"""

import sqlite3
import pandas as pd
from datetime import datetime

class FixedMLBAnalyzer:
    def __init__(self, db_path='mlb_historical_data_10_years.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def get_top_hitters(self, data_period='10_years', min_pa=500):
        """Get top hitters by wOBA."""
        query = """
        SELECT 
            player_name,
            player_id,
            ba,
            obp,
            slg,
            woba,
            xwoba,
            pa,
            launch_speed,
            hardhit_percent,
            barrels_per_bbe_percent
        FROM savant_batting_stats
        WHERE data_period = ? 
            AND pa >= ?
            AND woba IS NOT NULL
        ORDER BY woba DESC
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, self.conn, params=[data_period, min_pa])
        return df
        
    def get_breakout_candidates(self, data_period='3_years', min_pa=200):
        """Identify breakout candidates based on expected stats."""
        query = """
        SELECT 
            player_name,
            player_id,
            ba,
            xba,
            (xba - ba) as ba_upside,
            woba,
            xwoba,
            (xwoba - woba) as woba_upside,
            obp,
            xobp,
            (xobp - obp) as obp_upside,
            slg,
            xslg,
            (xslg - slg) as slg_upside,
            launch_speed,
            hardhit_percent,
            barrels_per_bbe_percent,
            pa
        FROM savant_batting_stats
        WHERE data_period = ?
            AND pa >= ?
            AND xba > ba
            AND xwoba > woba
        ORDER BY (xwoba - woba) DESC
        LIMIT 25
        """
        
        df = pd.read_sql_query(query, self.conn, params=[data_period, min_pa])
        return df
        
    def get_regression_candidates(self, data_period='3_years', min_pa=200):
        """Identify regression candidates."""
        query = """
        SELECT 
            player_name,
            player_id,
            ba,
            xba,
            (ba - xba) as ba_overperformance,
            woba,
            xwoba,
            (woba - xwoba) as woba_overperformance,
            obp,
            xobp,
            slg,
            xslg,
            babip,
            launch_speed,
            hardhit_percent,
            pa
        FROM savant_batting_stats
        WHERE data_period = ?
            AND pa >= ?
            AND ba > xba
            AND woba > xwoba
            AND babip > 0.320
        ORDER BY (woba - xwoba) DESC
        LIMIT 25
        """
        
        df = pd.read_sql_query(query, self.conn, params=[data_period, min_pa])
        return df
        
    def get_top_pitchers(self, data_period='10_years', min_pitches=2000):
        """Get top pitchers by various metrics."""
        query = """
        SELECT 
            player_name,
            player_id,
            velocity,
            spin_rate,
            k_percent,
            bb_percent,
            (k_percent - bb_percent) as k_minus_bb,
            woba,
            xwoba,
            hardhit_percent,
            barrels_per_bbe_percent,
            pitches
        FROM savant_pitching_stats
        WHERE data_period = ?
            AND pitches >= ?
            AND k_percent IS NOT NULL
        ORDER BY k_percent DESC
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, self.conn, params=[data_period, min_pitches])
        return df
        
    def close(self):
        if self.conn:
            self.conn.close()

def main():
    analyzer = FixedMLBAnalyzer()
    
    print("🏆 TOP HITTERS (10-Year wOBA)")
    print("=" * 50)
    top_hitters = analyzer.get_top_hitters('10_years', 500)
    print(top_hitters[['player_name', 'ba', 'obp', 'slg', 'woba', 'xwoba', 'pa']].to_string(index=False))
    
    print("\n🚀 BREAKOUT CANDIDATES (3-Year)")
    print("=" * 50)
    breakouts = analyzer.get_breakout_candidates('3_years', 200)
    if not breakouts.empty:
        print(breakouts[['player_name', 'ba', 'xba', 'ba_upside', 'woba', 'xwoba', 'woba_upside', 'pa']].head(10).to_string(index=False))
    
    print("\n📉 REGRESSION CANDIDATES (3-Year)")
    print("=" * 50)
    regressions = analyzer.get_regression_candidates('3_years', 200)
    if not regressions.empty:
        print(regressions[['player_name', 'ba', 'xba', 'ba_overperformance', 'woba', 'xwoba', 'woba_overperformance', 'pa']].head(10).to_string(index=False))
    
    print("\n⚾ TOP PITCHERS (10-Year K%)")
    print("=" * 50)
    top_pitchers = analyzer.get_top_pitchers('10_years', 2000)
    print(top_pitchers[['player_name', 'velocity', 'k_percent', 'bb_percent', 'k_minus_bb', 'woba', 'pitches']].head(10).to_string(index=False))
    
    analyzer.close()

if __name__ == "__main__":
    main()