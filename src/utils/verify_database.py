"""
Database Verification Script

Quick script to verify the database structure and data integrity
after importing Baseball Savant CSV files.
"""

import sqlite3
import pandas as pd

def verify_database():
    db_path = 'mlb_historical_data_10_years.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 DATABASE VERIFICATION REPORT")
        print("=" * 50)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"📊 Total Tables: {len(tables)}")
        print("\n🗂️  DETAILED TABLE INFORMATION:")
        
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
                
            print(f"\n📋 {table_name.upper()}")
            print("-" * 30)
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   Records: {count:,}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"   Columns: {len(columns)}")
            
            # Show sample data for Savant tables
            if 'savant' in table_name:
                print("   Sample Data:")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                
                if sample_data:
                    # Get column names
                    col_names = [col[1] for col in columns]
                    
                    for i, row in enumerate(sample_data):
                        print(f"     Row {i+1}:")
                        for j, (col_name, value) in enumerate(zip(col_names[:5], row[:5])):  # Show first 5 columns
                            print(f"       {col_name}: {value}")
                        print("       ...")
                        
            # Show data period breakdown for Savant tables
            if 'savant' in table_name and 'data_period' in [col[1] for col in columns]:
                print("   Data Periods:")
                cursor.execute(f"SELECT data_period, COUNT(*) FROM {table_name} GROUP BY data_period ORDER BY COUNT(*) DESC")
                periods = cursor.fetchall()
                for period, period_count in periods:
                    print(f"     {period}: {period_count:,} records")
                    
        # Verify data integrity
        print(f"\n🔍 DATA INTEGRITY CHECKS")
        print("-" * 30)
        
        # Check for null player_ids in Savant tables
        for table_type in ['batting', 'pitching']:
            table_name = f'savant_{table_type}_stats'
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE player_id IS NULL")
            null_count = cursor.fetchone()[0]
            print(f"   {table_name} - Null player_ids: {null_count}")
            
        # Check for duplicate player_id/data_period combinations
        for table_type in ['batting', 'pitching']:
            table_name = f'savant_{table_type}_stats'
            cursor.execute(f"""
                SELECT player_id, data_period, COUNT(*) as count 
                FROM {table_name} 
                GROUP BY player_id, data_period 
                HAVING COUNT(*) > 1
                LIMIT 5
            """)
            duplicates = cursor.fetchall()
            if duplicates:
                print(f"   {table_name} - Duplicate combinations found: {len(duplicates)}")
            else:
                print(f"   {table_name} - No duplicates found ✅")
                
        # Show top performers sample
        print(f"\n🏆 SAMPLE TOP PERFORMERS")
        print("-" * 30)
        
        # Top wOBA hitters (10-year period)
        cursor.execute("""
            SELECT player_name, woba, pa 
            FROM savant_batting_stats 
            WHERE data_period = '10_years' AND woba IS NOT NULL AND pa >= 1000
            ORDER BY woba DESC 
            LIMIT 5
        """)
        top_hitters = cursor.fetchall()
        
        print("   Top wOBA (10-year, min 1000 PA):")
        for i, (name, woba, pa) in enumerate(top_hitters, 1):
            print(f"     {i}. {name}: {woba:.3f} wOBA ({pa} PA)")
            
        # Top K% pitchers (10-year period)
        cursor.execute("""
            SELECT player_name, k_percent, pitches 
            FROM savant_pitching_stats 
            WHERE data_period = '10_years' AND k_percent IS NOT NULL AND pitches >= 5000
            ORDER BY k_percent DESC 
            LIMIT 5
        """)
        top_pitchers = cursor.fetchall()
        
        print("\n   Top K% (10-year, min 5000 pitches):")
        for i, (name, k_pct, pitches) in enumerate(top_pitchers, 1):
            print(f"     {i}. {name}: {k_pct:.1f}% K-rate ({pitches} pitches)")
            
        # Database file size
        import os
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"\n💾 Database file size: {size_mb:.1f} MB")
            
        conn.close()
        print(f"\n✅ Database verification completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify_database()