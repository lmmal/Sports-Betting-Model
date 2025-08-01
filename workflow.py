"""
MLB Betting Model Workflow

This script provides a complete workflow for the MLB betting model project:
1. Update historical training data
2. Retrain models with latest data
3. Generate features for today's games  
4. Make predictions and recommendations

Run this script daily to maintain and use the betting model.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.historical_data_collector import HistoricalMLBDataCollector
from modeling.betting_model_pipeline import MLBBettingModelPipeline  
from modeling.betting_predictor import MLBBettingPredictor


class MLBBettingWorkflow:
    def __init__(self):
        """Initialize the complete workflow"""
        self.collector = None
        self.pipeline = None
        self.predictor = None
        
    def collect_historical_data(self, days_back: int = 30, retrain: bool = False):
        """
        Collect historical data for training
        
        Args:
            days_back: Number of days back to collect
            retrain: Whether to retrain models after collecting data
        """
        print("🏆 STEP 1: COLLECTING HISTORICAL DATA")
        print("=" * 60)
        
        try:
            self.collector = HistoricalMLBDataCollector()
            
            # Collect recent games
            recent_df = self.collector.collect_recent_games(days_back=days_back)
            
            if not recent_df.empty:
                print(f"✅ Successfully collected {len(recent_df)} games")
                
                if retrain:
                    self.train_models()
                    
                return True
            else:
                print("❌ No historical data collected")
                return False
                
        except Exception as e:
            print(f"❌ Error collecting historical data: {e}")
            return False
        
        finally:
            if self.collector:
                self.collector.close()
    
    def train_models(self, data_path: str = None):
        """
        Train betting models
        
        Args:
            data_path: Path to training data (optional)
        """
        print("\n🏆 STEP 2: TRAINING BETTING MODELS")
        print("=" * 60)
        
        try:
            # Find training data
            if data_path is None:
                data_files = [
                    'data/recent_games_30days.parquet',
                    'data/historical_games.parquet'
                ]
                
                for file_path in data_files:
                    if os.path.exists(file_path):
                        data_path = file_path
                        break
                
                if data_path is None:
                    print("❌ No training data found")
                    return False
            
            # Initialize and train
            self.pipeline = MLBBettingModelPipeline(data_path)
            
            # Load and process data
            df = self.pipeline.load_data()
            features_df = self.pipeline.engineer_features(df)
            
            # Train models
            all_results = self.pipeline.train_all_models(features_df)
            
            # Save models
            self.pipeline.save_models()
            
            # Show performance
            self.pipeline.evaluate_betting_performance(all_results)
            
            print("✅ Models trained and saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def generate_todays_features(self):
        """
        Generate features for today's games
        """
        print("\n🏆 STEP 3: GENERATING TODAY'S FEATURES")
        print("=" * 60)
        
        try:
            # Import and run the feature builder
            from src.feature_engineering.streamlined_game_feature_builder import StreamlinedGameFeatureBuilder
            
            builder = StreamlinedGameFeatureBuilder()
            features_df = builder.build_game_features()
            
            if not features_df.empty:
                builder.save_features(features_df, 'games_features.parquet')
                print(f"✅ Generated features for {len(features_df)} games")
                return True
            else:
                print("⚠️ No games found for today")
                return False
                
        except Exception as e:
            print(f"❌ Error generating features: {e}")
            return False
    
    def make_predictions(self):
        """
        Make betting predictions for today's games
        """
        print("\n🏆 STEP 4: MAKING BETTING PREDICTIONS")
        print("=" * 60)
        
        try:
            # Check if models exist
            if not os.path.exists('models/model_metadata.joblib'):
                print("❌ No trained models found")
                print("💡 Run workflow with --train flag first")
                return False
            
            # Check if today's features exist
            if not os.path.exists('games_features.parquet'):
                print("❌ No feature data found")
                print("💡 Run workflow with --features flag first")
                return False
            
            # Initialize predictor and make predictions
            self.predictor = MLBBettingPredictor()
            predictions_df = self.predictor.predict_todays_games()
            
            if not predictions_df.empty:
                print("✅ Predictions generated successfully")
                return True
            else:
                print("⚠️ No predictions generated")
                return False
                
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return False
    
    def run_full_workflow(self, days_back: int = 30):
        """
        Run the complete workflow
        
        Args:
            days_back: Days of historical data to collect
        """
        print("🏆 MLB BETTING MODEL - COMPLETE WORKFLOW")
        print("=" * 80)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success_count = 0
        
        # Step 1: Collect historical data
        if self.collect_historical_data(days_back=days_back):
            success_count += 1
        
        # Step 2: Train models
        if self.train_models():
            success_count += 1
        
        # Step 3: Generate today's features
        if self.generate_todays_features():
            success_count += 1
        
        # Step 4: Make predictions
        if self.make_predictions():
            success_count += 1
        
        # Summary
        print(f"\n🎉 WORKFLOW COMPLETE")
        print("=" * 60)
        print(f"✅ Completed {success_count}/4 steps successfully")
        
        if success_count == 4:
            print("🏆 Full workflow successful!")
            print("📁 Check these files for results:")
            print("   • today_predictions.csv - Betting recommendations")
            print("   • today_predictions_full.csv - Detailed predictions")
        elif success_count >= 2:
            print("⚠️ Workflow partially successful")
        else:
            print("❌ Workflow failed")
        
        return success_count == 4


def main():
    """Main execution with command line arguments"""
    
    parser = argparse.ArgumentParser(description='MLB Betting Model Workflow')
    parser.add_argument('--collect', action='store_true', help='Collect historical data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--features', action='store_true', help='Generate today\'s features')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--full', action='store_true', help='Run full workflow')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = MLBBettingWorkflow()
    
    try:
        if args.full:
            # Run complete workflow
            workflow.run_full_workflow(days_back=args.days)
            
        else:
            # Run individual steps
            if args.collect:
                workflow.collect_historical_data(days_back=args.days)
            
            if args.train:
                workflow.train_models()
            
            if args.features:
                workflow.generate_todays_features()
            
            if args.predict:
                workflow.make_predictions()
            
            # If no flags specified, show help
            if not any([args.collect, args.train, args.features, args.predict]):
                print("🏆 MLB BETTING MODEL WORKFLOW")
                print("=" * 60)
                print("Available options:")
                print("  --collect    Collect historical training data")
                print("  --train      Train betting models")
                print("  --features   Generate features for today's games")
                print("  --predict    Make betting predictions")
                print("  --full       Run complete workflow")
                print("  --days N     Days of historical data (default: 30)")
                print()
                print("Examples:")
                print("  python workflow.py --full              # Complete workflow")
                print("  python workflow.py --collect --train   # Update models")
                print("  python workflow.py --features --predict # Daily predictions")
    
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
