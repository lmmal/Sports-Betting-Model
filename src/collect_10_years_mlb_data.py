"""
Collect 10 years of MLB historical data using the working MLB Stats API collector
"""

import os
import sys
from datetime import datetime

# Import our working collector
from working_mlb_api_collector import WorkingMLBAPICollector

def main():
    """Run the 10-year data collection."""
    print("🏆 MLB Historical Data Collection - 10 Years")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Remove the old 5-year database if it exists
    old_db = 'working_mlb_data.db'
    if os.path.exists(old_db):
        print(f"📁 Removing old database: {old_db}")
        os.remove(old_db)
    
    # Initialize collector for 10 years
    db_name = 'mlb_historical_data_10_years.db'
    collector = WorkingMLBAPICollector(db_path=db_name, years_back=10)
    
    print(f"📊 Configuration:")
    print(f"   Database: {db_name}")
    print(f"   Years: {collector.start_year} - {collector.current_year - 1}")
    print(f"   Total seasons: {collector.current_year - 1 - collector.start_year + 1}")
    print(f"   Expected records: ~{30 * (collector.current_year - 1 - collector.start_year + 1)} per table")
    
    try:
        # Run the full collection
        collector.collect_all_data()
        
        print(f"\n🎉 SUCCESS! 10-year MLB data collection completed!")
        print(f"📁 Database file: {os.path.abspath(db_name)}")
        
        # Show final file size
        if os.path.exists(db_name):
            size_mb = os.path.getsize(db_name) / (1024 * 1024)
            print(f"💾 Database size: {size_mb:.1f} MB")
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        collector.close()
        print(f"\n⏰ End time: {datetime.now()}")

if __name__ == "__main__":
    main()