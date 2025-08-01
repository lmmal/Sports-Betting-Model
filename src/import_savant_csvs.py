"""
Baseball Savant CSV Data Importer

This script imports all Baseball Savant CSV files into the existing MLB database,
creating comprehensive player-level statistics tables with Statcast data.

Features:
- Imports batting and pitching data from multiple time periods
- Creates properly structured tables with all Statcast metrics
- Handles data cleaning and type conversion
- Adds metadata about data sources and time periods
- Provides comprehensive logging and error handling

Author: Sports Betting Model
Date: 2024
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class BaseballSavantImporter:
    def __init__(self, db_path='mlb_historical_data_10_years.db'):
        """
        Initialize the Baseball Savant CSV Importer.
        
        Args:
            db_path (str): Path to existing SQLite database
        """
        self.db_path = db_path
        self.conn = None
        
        # Define CSV files and their corresponding time periods
        self.csv_files = {
            'batting': {
                'savant_batters_10.csv': '10_years',
                'savant_batters_5.csv': '5_years', 
                'savant_batters_3.csv': '3_years',
                'savant_batters_current.csv': 'current_year',
                'savant_batters_last.csv': 'last_year'
            },
            'pitching': {
                'savant_pitchers_10.csv': '10_years',
                'savant_pitchers_5.csv': '5_years',
                'savant_pitcher_3.csv': '3_years',  # Note: different naming
                'savant_pitchers_current.csv': 'current_year',
                'savant_pitchers_last.csv': 'last_year'
            }
        }
        
        # Connect to existing database
        self.connect_db()
        
    def connect_db(self):
        """Connect to the existing database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to existing database: {self.db_path}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            raise
            
    def create_savant_tables(self):
        """Create comprehensive tables for Baseball Savant data."""
        cursor = self.conn.cursor()
        
        print("📊 Creating Baseball Savant tables...")
        
        # Comprehensive batting statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS savant_batting_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                player_name TEXT,
                data_period TEXT,
                pitches INTEGER,
                total_pitches INTEGER,
                pitch_percent REAL,
                ba REAL,
                iso REAL,
                babip REAL,
                slg REAL,
                woba REAL,
                xwoba REAL,
                xba REAL,
                hits INTEGER,
                abs INTEGER,
                launch_speed REAL,
                launch_angle REAL,
                spin_rate REAL,
                velocity REAL,
                effective_speed REAL,
                whiffs INTEGER,
                swings INTEGER,
                takes INTEGER,
                eff_min_vel REAL,
                release_extension REAL,
                pos3_int_start_distance REAL,
                pos4_int_start_distance REAL,
                pos5_int_start_distance REAL,
                pos6_int_start_distance REAL,
                pos7_int_start_distance REAL,
                pos8_int_start_distance REAL,
                pos9_int_start_distance REAL,
                pitcher_run_exp REAL,
                run_exp REAL,
                bat_speed REAL,
                swing_length REAL,
                pa INTEGER,
                bip INTEGER,
                singles INTEGER,
                doubles INTEGER,
                triples INTEGER,
                hrs INTEGER,
                so INTEGER,
                k_percent REAL,
                bb INTEGER,
                bb_percent REAL,
                api_break_z_with_gravity REAL,
                api_break_z_induced REAL,
                api_break_x_arm REAL,
                api_break_x_batter_in REAL,
                hyper_speed REAL,
                bbdist INTEGER,
                hardhit_percent REAL,
                barrels_per_bbe_percent REAL,
                barrels_per_pa_percent REAL,
                release_pos_z REAL,
                release_pos_x REAL,
                plate_x REAL,
                plate_z REAL,
                obp REAL,
                barrels_total INTEGER,
                batter_run_value_per_100 REAL,
                xobp REAL,
                xslg REAL,
                pitcher_run_value_per_100 REAL,
                xbadiff REAL,
                xobpdiff REAL,
                xslgdiff REAL,
                wobadiff REAL,
                swing_miss_percent REAL,
                arm_angle REAL,
                attack_angle REAL,
                attack_direction REAL,
                swing_path_tilt REAL,
                rate_ideal_attack_angle REAL,
                intercept_ball_minus_batter_pos_x_inches REAL,
                intercept_ball_minus_batter_pos_y_inches REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, data_period)
            )
        ''')
        
        # Comprehensive pitching statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS savant_pitching_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                player_name TEXT,
                data_period TEXT,
                pitches INTEGER,
                total_pitches INTEGER,
                pitch_percent REAL,
                ba REAL,
                iso REAL,
                babip REAL,
                slg REAL,
                woba REAL,
                xwoba REAL,
                xba REAL,
                hits INTEGER,
                abs INTEGER,
                launch_speed REAL,
                launch_angle REAL,
                spin_rate REAL,
                velocity REAL,
                effective_speed REAL,
                whiffs INTEGER,
                swings INTEGER,
                takes INTEGER,
                eff_min_vel REAL,
                release_extension REAL,
                pos3_int_start_distance REAL,
                pos4_int_start_distance REAL,
                pos5_int_start_distance REAL,
                pos6_int_start_distance REAL,
                pos7_int_start_distance REAL,
                pos8_int_start_distance REAL,
                pos9_int_start_distance REAL,
                pitcher_run_exp REAL,
                run_exp REAL,
                bat_speed REAL,
                swing_length REAL,
                pa INTEGER,
                bip INTEGER,
                singles INTEGER,
                doubles INTEGER,
                triples INTEGER,
                hrs INTEGER,
                so INTEGER,
                k_percent REAL,
                bb INTEGER,
                bb_percent REAL,
                api_break_z_with_gravity REAL,
                api_break_z_induced REAL,
                api_break_x_arm REAL,
                api_break_x_batter_in REAL,
                hyper_speed REAL,
                bbdist INTEGER,
                hardhit_percent REAL,
                barrels_per_bbe_percent REAL,
                barrels_per_pa_percent REAL,
                release_pos_z REAL,
                release_pos_x REAL,
                plate_x REAL,
                plate_z REAL,
                obp REAL,
                barrels_total INTEGER,
                batter_run_value_per_100 REAL,
                xobp REAL,
                xslg REAL,
                pitcher_run_value_per_100 REAL,
                xbadiff REAL,
                xobpdiff REAL,
                xslgdiff REAL,
                wobadiff REAL,
                swing_miss_percent REAL,
                arm_angle REAL,
                attack_angle REAL,
                attack_direction REAL,
                swing_path_tilt REAL,
                rate_ideal_attack_angle REAL,
                intercept_ball_minus_batter_pos_x_inches REAL,
                intercept_ball_minus_batter_pos_y_inches REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, data_period)
            )
        ''')
        
        # Data source tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS savant_data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                data_type TEXT,
                data_period TEXT,
                records_imported INTEGER,
                import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size_bytes INTEGER,
                UNIQUE(file_name, data_type, data_period)
            )
        ''')
        
        self.conn.commit()
        print("✅ Baseball Savant tables created successfully!")
        
    def clean_numeric_columns(self, df):
        """Clean and convert numeric columns, handling various data issues."""
        
        # List of columns that should be numeric
        numeric_columns = [
            'pitches', 'total_pitches', 'pitch_percent', 'ba', 'iso', 'babip', 'slg', 'woba', 
            'xwoba', 'xba', 'hits', 'abs', 'launch_speed', 'launch_angle', 'spin_rate', 
            'velocity', 'effective_speed', 'whiffs', 'swings', 'takes', 'eff_min_vel',
            'release_extension', 'pos3_int_start_distance', 'pos4_int_start_distance',
            'pos5_int_start_distance', 'pos6_int_start_distance', 'pos7_int_start_distance',
            'pos8_int_start_distance', 'pos9_int_start_distance', 'pitcher_run_exp', 'run_exp',
            'bat_speed', 'swing_length', 'pa', 'bip', 'singles', 'doubles', 'triples', 'hrs',
            'so', 'k_percent', 'bb', 'bb_percent', 'api_break_z_with_gravity', 
            'api_break_z_induced', 'api_break_x_arm', 'api_break_x_batter_in', 'hyper_speed',
            'bbdist', 'hardhit_percent', 'barrels_per_bbe_percent', 'barrels_per_pa_percent',
            'release_pos_z', 'release_pos_x', 'plate_x', 'plate_z', 'obp', 'barrels_total',
            'batter_run_value_per_100', 'xobp', 'xslg', 'pitcher_run_value_per_100',
            'xbadiff', 'xobpdiff', 'xslgdiff', 'wobadiff', 'swing_miss_percent', 'arm_angle',
            'attack_angle', 'attack_direction', 'swing_path_tilt', 'rate_ideal_attack_angle',
            'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Handle string values that might be in quotes
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN values with 0 for integer columns, keep NaN for float columns
                if col in ['pitches', 'total_pitches', 'hits', 'abs', 'whiffs', 'swings', 'takes',
                          'pa', 'bip', 'singles', 'doubles', 'triples', 'hrs', 'so', 'bb', 
                          'bbdist', 'barrels_total']:
                    df[col] = df[col].fillna(0).astype('Int64')  # Nullable integer
                else:
                    df[col] = df[col].astype('float64')  # Keep as float, NaN allowed
        
        return df
        
    def import_csv_file(self, file_path, data_type, data_period):
        """Import a single CSV file into the database."""
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return 0
            
        try:
            print(f"📥 Importing {data_type} data from {os.path.basename(file_path)} ({data_period})...")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"⚠️  Empty file: {file_path}")
                return 0
                
            # Add metadata columns
            df['data_period'] = data_period
            
            # Clean column names (remove quotes if present)
            df.columns = df.columns.str.replace('"', '')
            
            # Clean numeric data
            df = self.clean_numeric_columns(df)
            
            # Clean player names (remove quotes)
            if 'player_name' in df.columns:
                df['player_name'] = df['player_name'].astype(str).str.replace('"', '')
            
            # Determine target table
            table_name = f'savant_{data_type}_stats'
            
            # Insert data into database
            df.to_sql(table_name, self.conn, if_exists='append', index=False)
            
            # Record the import in data sources table
            file_size = os.path.getsize(file_path)
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO savant_data_sources 
                (file_name, data_type, data_period, records_imported, file_size_bytes)
                VALUES (?, ?, ?, ?, ?)
            ''', (os.path.basename(file_path), data_type, data_period, len(df), file_size))
            
            self.conn.commit()
            
            print(f"✅ Imported {len(df):,} {data_type} records from {data_period} period")
            return len(df)
            
        except Exception as e:
            print(f"❌ Error importing {file_path}: {e}")
            return 0
            
    def import_all_csvs(self):
        """Import all Baseball Savant CSV files."""
        print("🚀 Starting Baseball Savant CSV import process")
        print("=" * 60)
        
        # Create tables first
        self.create_savant_tables()
        
        total_records = 0
        
        # Import batting data
        print("\n🏏 IMPORTING BATTING DATA")
        print("-" * 30)
        batting_records = 0
        
        for filename, period in self.csv_files['batting'].items():
            file_path = os.path.join(os.path.dirname(self.db_path), filename)
            records = self.import_csv_file(file_path, 'batting', period)
            batting_records += records
            total_records += records
            
        # Import pitching data  
        print("\n⚾ IMPORTING PITCHING DATA")
        print("-" * 30)
        pitching_records = 0
        
        for filename, period in self.csv_files['pitching'].items():
            file_path = os.path.join(os.path.dirname(self.db_path), filename)
            records = self.import_csv_file(file_path, 'pitching', period)
            pitching_records += records
            total_records += records
            
        # Final summary
        print(f"\n🎉 IMPORT COMPLETE!")
        print("=" * 60)
        print(f"📊 Total Records Imported:")
        print(f"   🏏 Batting: {batting_records:,}")
        print(f"   ⚾ Pitching: {pitching_records:,}")
        print(f"   📈 Total: {total_records:,}")
        
        # Show database size
        if os.path.exists(self.db_path):
            size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            print(f"💾 Database size: {size_mb:.1f} MB")
            
        # Show table counts
        self.show_table_summary()
        
    def show_table_summary(self):
        """Display summary of imported data."""
        print(f"\n📋 DATABASE SUMMARY")
        print("-" * 30)
        
        cursor = self.conn.cursor()
        
        # Get table counts
        tables_to_check = [
            'savant_batting_stats',
            'savant_pitching_stats', 
            'savant_data_sources'
        ]
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count:,} records")
            except:
                print(f"   {table}: Table not found")
                
        # Show data periods breakdown
        print(f"\n📅 DATA PERIODS BREAKDOWN")
        print("-" * 30)
        
        for data_type in ['batting', 'pitching']:
            print(f"\n{data_type.title()} Data:")
            try:
                cursor.execute(f'''
                    SELECT data_period, COUNT(*) as count 
                    FROM savant_{data_type}_stats 
                    GROUP BY data_period 
                    ORDER BY count DESC
                ''')
                results = cursor.fetchall()
                for period, count in results:
                    print(f"   {period}: {count:,} players")
            except:
                print(f"   No {data_type} data found")
                
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")

def main():
    """Main execution function."""
    print("🏆 Baseball Savant CSV Data Importer")
    print("=" * 60)
    print(f"📅 Data Periods: 10yr, 5yr, 3yr, current, last year")
    print(f"📊 Data Types: Batting & Pitching Statcast metrics")
    print(f"⏰ Start time: {datetime.now()}")
    
    importer = None
    
    try:
        # Initialize importer
        importer = BaseballSavantImporter()
        
        # Import all CSV files
        importer.import_all_csvs()
        
        print(f"\n✅ SUCCESS! Baseball Savant data import completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Import interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if importer:
            importer.close()
        print(f"\n⏰ End time: {datetime.now()}")

if __name__ == "__main__":
    main()