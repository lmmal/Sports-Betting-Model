"""
MLB Betting Predictions

This script loads trained models and makes predictions on today's games.
Uses the current games_features.parquet file and trained models to generate:
1. Win probability predictions
2. Over/under predictions  
3. Spread coverage predictions
4. Score predictions

Output: today_predictions.csv with betting recommendations
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas.game_features_schema import MLBGameFeatures

import warnings
warnings.filterwarnings('ignore')


class MLBBettingPredictor:
    def __init__(self, model_dir: str = 'models/'):
        """
        Initialize the betting predictor
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_columns = []
        self.encoders = {}
        self.model_metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and metadata"""
        print(f"📁 Loading models from: {self.model_dir}")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'model_metadata.joblib')
        if os.path.exists(metadata_path):
            self.model_metadata = joblib.load(metadata_path)
            self.feature_columns = self.model_metadata['feature_columns']
            self.encoders = self.model_metadata['encoders']
            print(f"✅ Loaded model metadata")
        else:
            print(f"⚠️ No model metadata found at {metadata_path}")
            return
        
        # Load individual models
        model_files = {
            'home_win': 'home_win_model.joblib',
            'over_under': 'over_under_model.joblib', 
            'spread': 'spread_model.joblib',
            'total_runs': 'total_runs_model.joblib',
            'home_score': 'home_score_model.joblib',
            'away_score': 'away_score_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded {model_name} model")
            else:
                print(f"⚠️ Model not found: {filename}")
        
        print(f"✅ Loaded {len(self.models)} models")
    
    def load_todays_games(self, games_file: str = 'games_features.parquet') -> pd.DataFrame:
        """
        Load today's games features
        
        Args:
            games_file: Path to games features file
            
        Returns:
            DataFrame with today's games
        """
        print(f"📊 Loading today's games from: {games_file}")
        
        if not os.path.exists(games_file):
            print(f"❌ Games file not found: {games_file}")
            print("💡 Run streamlined_game_feature_builder.py first")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(games_file)
            print(f"✅ Loaded {len(df)} games for prediction")
            return df
        except Exception as e:
            print(f"❌ Error loading games: {e}")
            return pd.DataFrame()
    
    def engineer_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for prediction (same as training)
        
        Args:
            df: Raw games data
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Engineering prediction features...")
        
        features_df = df.copy()
        
        # Convert date if needed
        if 'game_date' in features_df.columns:
            features_df['game_date'] = pd.to_datetime(features_df['game_date'])
            
            # Extract date features
            features_df['day_of_week'] = features_df['game_date'].dt.dayofweek
            features_df['month'] = features_df['game_date'].dt.month
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables using existing encoders
        categorical_columns = ['home_team', 'away_team']
        for col in categorical_columns:
            if col in features_df.columns and col in self.encoders:
                le = self.encoders[col]
                # Handle unseen categories
                features_df[f'{col}_encoded'] = features_df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Create spread differential features
        if 'home_spread' in features_df.columns and 'away_spread' in features_df.columns:
            features_df['spread_diff'] = features_df['home_spread'] - features_df['away_spread']
        
        # Create odds features
        if 'home_ml' in features_df.columns and 'away_ml' in features_df.columns:
            features_df['home_ml_prob'] = self.american_odds_to_prob(features_df['home_ml'])
            features_df['away_ml_prob'] = self.american_odds_to_prob(features_df['away_ml'])
            features_df['ml_prob_diff'] = features_df['home_ml_prob'] - features_df['away_ml_prob']
        
        print(f"✅ Engineered features for prediction")
        return features_df
    
    def american_odds_to_prob(self, odds: pd.Series) -> pd.Series:
        """Convert American odds to implied probability"""
        prob = np.where(odds > 0, 
                       100 / (odds + 100),
                       -odds / (-odds + 100))
        return pd.Series(prob, index=odds.index)
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare data for prediction
        
        Args:
            df: Engineered features DataFrame
            
        Returns:
            Feature matrix for prediction
        """
        # Get only the features used in training
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        # Create feature matrix with missing features as 0
        X = np.zeros((len(df), len(self.feature_columns)))
        
        for i, feature in enumerate(self.feature_columns):
            if feature in df.columns:
                X[:, i] = df[feature].fillna(0)
        
        return X
    
    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make betting predictions for games
        
        Args:
            df: Games DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        print("🎯 Making betting predictions...")
        
        if df.empty:
            print("❌ No games to predict")
            return pd.DataFrame()
        
        # Engineer features
        features_df = self.engineer_prediction_features(df)
        
        # Prepare prediction data
        X = self.prepare_prediction_data(features_df)
        
        # Initialize predictions DataFrame
        predictions_df = df[['game_date', 'home_team', 'away_team']].copy()
        
        # Add odds for reference
        odds_columns = ['home_ml', 'away_ml', 'home_spread', 'away_spread', 'over_under']
        for col in odds_columns:
            if col in df.columns:
                predictions_df[col] = df[col]
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Classification - get probabilities
                    proba = model.predict_proba(X)
                    predictions_df[f'{model_name}_prob'] = proba[:, 1]
                    predictions_df[f'{model_name}_pred'] = model.predict(X)
                else:
                    # Regression - get predictions
                    pred = model.predict(X)
                    predictions_df[f'{model_name}_pred'] = pred
                
                print(f"✅ Generated {model_name} predictions")
                
            except Exception as e:
                print(f"❌ Error with {model_name} model: {e}")
        
        return predictions_df
    
    def generate_betting_recommendations(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate betting recommendations based on predictions
        
        Args:
            predictions_df: DataFrame with model predictions
            
        Returns:
            DataFrame with betting recommendations
        """
        print("💡 Generating betting recommendations...")
        
        recommendations = []
        
        for _, game in predictions_df.iterrows():
            game_rec = {
                'game_date': game['game_date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'recommendations': []
            }
            
            # Home win recommendation
            if 'home_win_prob' in game:
                home_win_prob = game['home_win_prob']
                
                # Calculate implied probability from moneyline
                if 'home_ml' in game and pd.notna(game['home_ml']):
                    home_ml_implied = self.american_odds_to_prob(pd.Series([game['home_ml']]))[0]
                    
                    # Recommend if our model is significantly different from market
                    if home_win_prob > home_ml_implied + 0.1:  # 10% edge
                        game_rec['recommendations'].append({
                            'bet_type': 'moneyline',
                            'team': game['home_team'],
                            'confidence': 'high',
                            'model_prob': home_win_prob,
                            'market_prob': home_ml_implied,
                            'edge': home_win_prob - home_ml_implied
                        })
                    elif home_win_prob < home_ml_implied - 0.1:
                        game_rec['recommendations'].append({
                            'bet_type': 'moneyline',
                            'team': game['away_team'],
                            'confidence': 'high',
                            'model_prob': 1 - home_win_prob,
                            'market_prob': 1 - home_ml_implied,
                            'edge': (1 - home_win_prob) - (1 - home_ml_implied)
                        })
            
            # Over/under recommendation
            if 'over_under_prob' in game:
                over_prob = game['over_under_prob']
                
                if over_prob > 0.6:  # High confidence over
                    game_rec['recommendations'].append({
                        'bet_type': 'total',
                        'pick': 'over',
                        'confidence': 'medium' if over_prob < 0.7 else 'high',
                        'model_prob': over_prob
                    })
                elif over_prob < 0.4:  # High confidence under
                    game_rec['recommendations'].append({
                        'bet_type': 'total',
                        'pick': 'under',
                        'confidence': 'medium' if over_prob > 0.3 else 'high',
                        'model_prob': 1 - over_prob
                    })
            
            # Spread recommendation
            if 'spread_prob' in game:
                spread_prob = game['spread_prob']
                
                if spread_prob > 0.6:  # Home covers
                    game_rec['recommendations'].append({
                        'bet_type': 'spread',
                        'team': game['home_team'],
                        'confidence': 'medium' if spread_prob < 0.7 else 'high',
                        'model_prob': spread_prob
                    })
                elif spread_prob < 0.4:  # Away covers
                    game_rec['recommendations'].append({
                        'bet_type': 'spread',
                        'team': game['away_team'],
                        'confidence': 'medium' if spread_prob > 0.3 else 'high',
                        'model_prob': 1 - spread_prob
                    })
            
            recommendations.append(game_rec)
        
        # Convert to flat DataFrame
        flat_recommendations = []
        for game_rec in recommendations:
            if game_rec['recommendations']:
                for rec in game_rec['recommendations']:
                    flat_rec = {
                        'game_date': game_rec['game_date'],
                        'home_team': game_rec['home_team'],
                        'away_team': game_rec['away_team'],
                        **rec
                    }
                    flat_recommendations.append(flat_rec)
        
        rec_df = pd.DataFrame(flat_recommendations)
        print(f"✅ Generated {len(rec_df)} betting recommendations")
        
        return rec_df
    
    def predict_todays_games(self, 
                           games_file: str = 'games_features.parquet',
                           output_file: str = 'today_predictions.csv') -> pd.DataFrame:
        """
        Complete prediction pipeline for today's games
        
        Args:
            games_file: Input games features file
            output_file: Output predictions file
            
        Returns:
            DataFrame with predictions and recommendations
        """
        print("🏆 MLB BETTING PREDICTIONS")
        print("=" * 50)
        print(f"📅 Prediction date: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Load today's games
        games_df = self.load_todays_games(games_file)
        if games_df.empty:
            return pd.DataFrame()
        
        # Make predictions
        predictions_df = self.make_predictions(games_df)
        
        # Generate recommendations
        recommendations_df = self.generate_betting_recommendations(predictions_df)
        
        # Save predictions
        try:
            predictions_df.to_csv(output_file.replace('.csv', '_full.csv'), index=False)
            if not recommendations_df.empty:
                recommendations_df.to_csv(output_file, index=False)
                print(f"✅ Saved predictions: {output_file}")
                
                # Show summary
                print(f"\n📊 PREDICTION SUMMARY:")
                print(f"   • Games analyzed: {len(predictions_df)}")
                print(f"   • Betting recommendations: {len(recommendations_df)}")
                if not recommendations_df.empty:
                    print(f"   • High confidence bets: {len(recommendations_df[recommendations_df['confidence'] == 'high'])}")
                    print(f"   • Medium confidence bets: {len(recommendations_df[recommendations_df['confidence'] == 'medium'])}")
            else:
                print("⚠️ No betting recommendations generated")
                
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
        
        return predictions_df


def main():
    """Main execution function"""
    
    print("🏆 MLB BETTING PREDICTOR")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Check if models exist
        if not os.path.exists('models/model_metadata.joblib'):
            print("❌ No trained models found")
            print("💡 Run betting_model_pipeline.py first to train models")
            return
        
        # Initialize predictor
        predictor = MLBBettingPredictor()
        
        # Make predictions
        predictions_df = predictor.predict_todays_games()
        
        if not predictions_df.empty:
            print(f"\n🎉 SUCCESS! Betting predictions generated")
            print(f"   • Check today_predictions.csv for recommendations")
            print(f"   • Check today_predictions_full.csv for detailed predictions")
        else:
            print("⚠️ No predictions generated")
        
    except KeyboardInterrupt:
        print("\n⚠️ Prediction interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
