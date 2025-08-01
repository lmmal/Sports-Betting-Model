"""
MLB Betting Model Training Pipeline

This script implements a first-cut ML modeling pipeline for MLB betting:
1. Loads historical game data with outcomes
2. Engineers features from the data
3. Trains multiple models (Logistic Regression, Gradient Boosting)
4. Evaluates model performance on betting metrics
5. Saves trained models for prediction

Models trained:
- Home team win probability (classification)
- Over/under probability (classification) 
- Home team spread coverage (classification)
- Score prediction (regression)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas.game_features_schema import MLBGameFeatures, get_feature_names, get_target_names

warnings.filterwarnings('ignore')


class MLBBettingModelPipeline:
    def __init__(self, data_path: str = None):
        """
        Initialize the MLB betting model pipeline
        
        Args:
            data_path: Path to historical games parquet file
        """
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = get_target_names()
        self.encoders = {}
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        }
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load historical games data
        
        Args:
            data_path: Path to parquet file
            
        Returns:
            DataFrame with historical game data
        """
        if data_path is None:
            data_path = self.data_path
            
        if data_path is None:
            raise ValueError("No data path provided")
        
        print(f"📊 Loading data from: {data_path}")
        
        try:
            df = pd.read_parquet(data_path)
            print(f"✅ Loaded {len(df)} historical games")
            
            # Show data summary
            print(f"\n📈 DATA SUMMARY:")
            print(f"   • Date range: {df['game_date'].min()} to {df['game_date'].max()}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Missing values: {df.isnull().sum().sum()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models
        
        Args:
            df: Raw historical data
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Engineering features...")
        
        features_df = df.copy()
        
        # Convert date to datetime
        features_df['game_date'] = pd.to_datetime(features_df['game_date'])
        
        # Extract date features
        features_df['day_of_week'] = features_df['game_date'].dt.dayofweek
        features_df['month'] = features_df['game_date'].dt.month
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['home_team', 'away_team']
        for col in categorical_columns:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f'{col}_encoded'] = le.fit_transform(features_df[col])
                self.encoders[col] = le
        
        # Create spread differential features
        if 'home_spread' in features_df.columns and 'away_spread' in features_df.columns:
            features_df['spread_diff'] = features_df['home_spread'] - features_df['away_spread']
        
        # Create odds features
        if 'home_ml' in features_df.columns and 'away_ml' in features_df.columns:
            # Convert American odds to implied probability
            features_df['home_ml_prob'] = self.american_odds_to_prob(features_df['home_ml'])
            features_df['away_ml_prob'] = self.american_odds_to_prob(features_df['away_ml'])
            features_df['ml_prob_diff'] = features_df['home_ml_prob'] - features_df['away_ml_prob']
        
        # Select numeric features for modeling
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns 
                          if col not in ['game_id', 'home_score', 'away_score', 'total_runs']]
        
        self.feature_columns = feature_columns
        
        print(f"✅ Engineered {len(feature_columns)} features")
        return features_df
    
    def american_odds_to_prob(self, odds: pd.Series) -> pd.Series:
        """Convert American odds to implied probability"""
        prob = np.where(odds > 0, 
                       100 / (odds + 100),
                       -odds / (-odds + 100))
        return pd.Series(prob, index=odds.index)
    
    def prepare_model_data(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data for different model types
        
        Args:
            df: Engineered features DataFrame
            
        Returns:
            Dictionary with X, y arrays for each model type
        """
        print("🎯 Preparing model data...")
        
        # Get feature matrix
        X = df[self.feature_columns].fillna(0)
        
        model_data = {}
        
        # Classification targets
        if 'home_won' in df.columns:
            model_data['home_win'] = (X, df['home_won'].astype(int))
        
        if 'over_hit' in df.columns:
            model_data['over_under'] = (X, df['over_hit'].astype(int))
        
        if 'home_covered' in df.columns:
            model_data['spread'] = (X, df['home_covered'].astype(int))
        
        # Regression targets
        if 'total_runs' in df.columns:
            model_data['total_runs'] = (X, df['total_runs'])
        
        if 'home_score' in df.columns:
            model_data['home_score'] = (X, df['home_score'])
        
        if 'away_score' in df.columns:
            model_data['away_score'] = (X, df['away_score'])
        
        print(f"✅ Prepared data for {len(model_data)} model types")
        return model_data
    
    def train_classification_models(self, X: np.ndarray, y: np.ndarray, 
                                  target_name: str) -> Dict[str, Any]:
        """
        Train classification models for a target
        
        Args:
            X: Feature matrix
            y: Target vector
            target_name: Name of target variable
            
        Returns:
            Dictionary with trained models and metrics
        """
        print(f"🤖 Training classification models for: {target_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {}
        results = {}
        
        # Logistic Regression
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**self.model_configs['logistic_regression']))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_pred = lr_pipeline.predict(X_test)
        lr_pred_proba = lr_pipeline.predict_proba(X_test)[:, 1]
        
        models['logistic_regression'] = lr_pipeline
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_pred_proba)
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(**self.model_configs['gradient_boosting'])
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
        
        models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test, gb_pred),
            'precision': precision_score(y_test, gb_pred),
            'recall': recall_score(y_test, gb_pred),
            'f1': f1_score(y_test, gb_pred),
            'auc': roc_auc_score(y_test, gb_pred_proba)
        }
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
        best_model = models[best_model_name]
        
        print(f"✅ Best model for {target_name}: {best_model_name} (AUC: {results[best_model_name]['auc']:.3f})")
        
        return {
            'models': models,
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name
        }
    
    def train_regression_models(self, X: np.ndarray, y: np.ndarray, 
                              target_name: str) -> Dict[str, Any]:
        """
        Train regression models for a target
        
        Args:
            X: Feature matrix
            y: Target vector  
            target_name: Name of target variable
            
        Returns:
            Dictionary with trained models and metrics
        """
        print(f"📊 Training regression models for: {target_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {}
        results = {}
        
        # Linear Regression
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_pred = lr_pipeline.predict(X_test)
        
        models['linear_regression'] = lr_pipeline
        results['linear_regression'] = {
            'mse': mean_squared_error(y_test, lr_pred),
            'mae': mean_absolute_error(y_test, lr_pred),
            'r2': r2_score(y_test, lr_pred)
        }
        
        # Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        
        models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'mse': mean_squared_error(y_test, gb_pred),
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred)
        }
        
        # Select best model (highest R2)
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = models[best_model_name]
        
        print(f"✅ Best model for {target_name}: {best_model_name} (R²: {results[best_model_name]['r2']:.3f})")
        
        return {
            'models': models,
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models for different betting targets
        
        Args:
            df: Engineered features DataFrame
            
        Returns:
            Dictionary with all trained models
        """
        print("🏆 TRAINING ALL BETTING MODELS")
        print("=" * 50)
        
        # Prepare data
        model_data = self.prepare_model_data(df)
        all_results = {}
        
        # Classification models
        classification_targets = ['home_win', 'over_under', 'spread']
        for target in classification_targets:
            if target in model_data:
                X, y = model_data[target]
                results = self.train_classification_models(X, y, target)
                all_results[target] = results
                self.models[target] = results['best_model']
        
        # Regression models
        regression_targets = ['total_runs', 'home_score', 'away_score']
        for target in regression_targets:
            if target in model_data:
                X, y = model_data[target]
                results = self.train_regression_models(X, y, target)
                all_results[target] = results
                self.models[target] = results['best_model']
        
        return all_results
    
    def save_models(self, model_dir: str = 'models/'):
        """
        Save trained models to disk
        
        Args:
            model_dir: Directory to save models
        """
        print(f"💾 Saving models to: {model_dir}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        for target_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{target_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"✅ Saved {target_name} model")
        
        # Save feature columns and encoders
        metadata = {
            'feature_columns': self.feature_columns,
            'encoders': self.encoders,
            'model_configs': self.model_configs,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"✅ Saved model metadata")
    
    def evaluate_betting_performance(self, all_results: Dict[str, Any]) -> None:
        """
        Print detailed evaluation of betting model performance
        
        Args:
            all_results: Results from all trained models
        """
        print("\n📈 BETTING MODEL PERFORMANCE")
        print("=" * 60)
        
        for target_name, results in all_results.items():
            print(f"\n🎯 {target_name.upper()} PREDICTION:")
            print(f"   Best Model: {results['best_model_name']}")
            
            best_results = results['results'][results['best_model_name']]
            
            if 'auc' in best_results:  # Classification
                print(f"   • Accuracy: {best_results['accuracy']:.3f}")
                print(f"   • Precision: {best_results['precision']:.3f}")
                print(f"   • Recall: {best_results['recall']:.3f}")
                print(f"   • F1 Score: {best_results['f1']:.3f}")
                print(f"   • AUC: {best_results['auc']:.3f}")
            else:  # Regression
                print(f"   • R² Score: {best_results['r2']:.3f}")
                print(f"   • MAE: {best_results['mae']:.3f}")
                print(f"   • RMSE: {np.sqrt(best_results['mse']):.3f}")


def main():
    """Main execution function"""
    
    print("🏆 MLB BETTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        pipeline = MLBBettingModelPipeline()
        
        # Try to find historical data
        data_files = [
            'data/recent_games_30days.parquet',
            'data/historical_games.parquet',
            'historical_games.parquet'
        ]
        
        data_path = None
        for file_path in data_files:
            if os.path.exists(file_path):
                data_path = file_path
                break
        
        if data_path is None:
            print("❌ No historical data found. Please run historical_data_collector.py first")
            print("💡 Expected files:")
            for file_path in data_files:
                print(f"   • {file_path}")
            return
        
        # Load and process data
        df = pipeline.load_data(data_path)
        features_df = pipeline.engineer_features(df)
        
        # Train models
        all_results = pipeline.train_all_models(features_df)
        
        # Save models
        pipeline.save_models()
        
        # Evaluate performance
        pipeline.evaluate_betting_performance(all_results)
        
        print(f"\n🎉 SUCCESS! Betting models trained and saved")
        print(f"   • Models saved in: models/")
        print(f"   • Ready for predictions on new games")
        print(f"   • Use the models to predict: win probability, over/under, spread coverage")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
