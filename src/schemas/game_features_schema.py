"""
MLB Game Features Schema

This module defines the data contract for game features using Pydantic.
This ensures type safety and prevents silent schema changes that could break the model.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class Team(str, Enum):
    """MLB Team abbreviations"""
    ARI = "ARI"
    ATL = "ATL" 
    BAL = "BAL"
    BOS = "BOS"
    CHC = "CHC"
    CWS = "CWS"
    CIN = "CIN"
    CLE = "CLE"
    COL = "COL"
    DET = "DET"
    HOU = "HOU"
    KC = "KC"
    LAA = "LAA"
    LAD = "LAD"
    MIA = "MIA"
    MIL = "MIL"
    MIN = "MIN"
    NYM = "NYM"
    NYY = "NYY"
    OAK = "OAK"
    PHI = "PHI"
    PIT = "PIT"
    SD = "SD"
    SF = "SF"
    SEA = "SEA"
    STL = "STL"
    TB = "TB"
    TEX = "TEX"
    TOR = "TOR"
    WSH = "WSH"


class OddsData(BaseModel):
    """Betting odds for a game"""
    home_ml: Optional[float] = Field(None, description="Home team moneyline odds")
    away_ml: Optional[float] = Field(None, description="Away team moneyline odds")
    home_spread: Optional[float] = Field(None, description="Home team spread")
    away_spread: Optional[float] = Field(None, description="Away team spread")
    over_under: Optional[float] = Field(None, description="Over/under total")
    over_odds: Optional[float] = Field(None, description="Over odds")
    under_odds: Optional[float] = Field(None, description="Under odds")


class TeamStats(BaseModel):
    """Team performance statistics"""
    wins: Optional[int] = Field(None, description="Season wins")
    losses: Optional[int] = Field(None, description="Season losses")
    win_pct: Optional[float] = Field(None, description="Win percentage")
    runs_scored_avg: Optional[float] = Field(None, description="Average runs scored per game")
    runs_allowed_avg: Optional[float] = Field(None, description="Average runs allowed per game")
    era: Optional[float] = Field(None, description="Team ERA")
    whip: Optional[float] = Field(None, description="Team WHIP")
    batting_avg: Optional[float] = Field(None, description="Team batting average")
    obp: Optional[float] = Field(None, description="Team on-base percentage")
    slg: Optional[float] = Field(None, description="Team slugging percentage")
    ops: Optional[float] = Field(None, description="Team OPS")
    home_record: Optional[str] = Field(None, description="Home record (W-L)")
    away_record: Optional[str] = Field(None, description="Away record (W-L)")
    last_10: Optional[str] = Field(None, description="Last 10 games record")
    streak: Optional[str] = Field(None, description="Current streak")


class PitcherStats(BaseModel):
    """Starting pitcher statistics"""
    name: Optional[str] = Field(None, description="Pitcher name")
    era: Optional[float] = Field(None, description="Pitcher ERA")
    whip: Optional[float] = Field(None, description="Pitcher WHIP")
    k_9: Optional[float] = Field(None, description="Strikeouts per 9 innings")
    bb_9: Optional[float] = Field(None, description="Walks per 9 innings")
    hr_9: Optional[float] = Field(None, description="Home runs per 9 innings")
    wins: Optional[int] = Field(None, description="Wins")
    losses: Optional[int] = Field(None, description="Losses")
    games_started: Optional[int] = Field(None, description="Games started")
    innings_pitched: Optional[float] = Field(None, description="Innings pitched")
    fip: Optional[float] = Field(None, description="Fielding Independent Pitching")
    babip: Optional[float] = Field(None, description="Batting Average on Balls in Play")


class PlayerAggregates(BaseModel):
    """Aggregated player statistics (Statcast data)"""
    avg_exit_velocity: Optional[float] = Field(None, description="Average exit velocity")
    max_exit_velocity: Optional[float] = Field(None, description="Maximum exit velocity")
    avg_launch_angle: Optional[float] = Field(None, description="Average launch angle")
    barrel_rate: Optional[float] = Field(None, description="Barrel rate percentage")
    hard_hit_rate: Optional[float] = Field(None, description="Hard hit rate percentage")
    xba: Optional[float] = Field(None, description="Expected batting average")
    xslg: Optional[float] = Field(None, description="Expected slugging percentage")
    xwoba: Optional[float] = Field(None, description="Expected weighted on-base average")


class GameOutcome(BaseModel):
    """Actual game outcome (for training data)"""
    home_score: Optional[int] = Field(None, description="Home team final score")
    away_score: Optional[int] = Field(None, description="Away team final score")
    winner: Optional[Team] = Field(None, description="Winning team")
    total_runs: Optional[int] = Field(None, description="Total runs scored")
    home_won: Optional[bool] = Field(None, description="Did home team win")
    over_hit: Optional[bool] = Field(None, description="Did total go over")
    home_covered: Optional[bool] = Field(None, description="Did home team cover spread")


class MLBGameFeatures(BaseModel):
    """Complete feature set for an MLB game"""
    
    # Game identification
    game_id: str = Field(..., description="Unique game identifier")
    game_date: date = Field(..., description="Game date")
    home_team: Team = Field(..., description="Home team")
    away_team: Team = Field(..., description="Away team")
    
    # Odds data
    odds: OddsData = Field(default_factory=OddsData, description="Betting odds")
    
    # Team statistics
    home_team_stats: TeamStats = Field(default_factory=TeamStats, description="Home team stats")
    away_team_stats: TeamStats = Field(default_factory=TeamStats, description="Away team stats")
    
    # Starting pitchers
    home_pitcher: PitcherStats = Field(default_factory=PitcherStats, description="Home starting pitcher")
    away_pitcher: PitcherStats = Field(default_factory=PitcherStats, description="Away starting pitcher")
    
    # Player aggregates
    home_player_aggregates: PlayerAggregates = Field(default_factory=PlayerAggregates, description="Home team player aggregates")
    away_player_aggregates: PlayerAggregates = Field(default_factory=PlayerAggregates, description="Away team player aggregates")
    
    # Game outcome (for historical/training data)
    outcome: Optional[GameOutcome] = Field(None, description="Game outcome (for training data)")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Feature creation timestamp")
    data_version: str = Field(default="1.0", description="Schema version")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields from being added


def validate_game_features(data: dict) -> MLBGameFeatures:
    """
    Validate and parse game features data
    
    Args:
        data: Dictionary of game features
        
    Returns:
        Validated MLBGameFeatures object
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    return MLBGameFeatures(**data)


def get_feature_names() -> List[str]:
    """
    Get list of all feature field names for ML model training
    
    Returns:
        List of feature field names
    """
    # Get all field names from the schema
    schema = MLBGameFeatures.schema()
    
    # Flatten nested fields for ML feature names
    feature_names = []
    
    # Basic fields
    feature_names.extend(['game_date', 'home_team', 'away_team'])
    
    # Odds fields
    odds_fields = ['home_ml', 'away_ml', 'home_spread', 'away_spread', 
                   'over_under', 'over_odds', 'under_odds']
    feature_names.extend([f'odds_{field}' for field in odds_fields])
    
    # Team stats fields  
    team_stat_fields = ['wins', 'losses', 'win_pct', 'runs_scored_avg', 'runs_allowed_avg',
                        'era', 'whip', 'batting_avg', 'obp', 'slg', 'ops']
    feature_names.extend([f'home_team_{field}' for field in team_stat_fields])
    feature_names.extend([f'away_team_{field}' for field in team_stat_fields])
    
    # Pitcher fields
    pitcher_fields = ['era', 'whip', 'k_9', 'bb_9', 'hr_9', 'wins', 'losses', 
                      'games_started', 'innings_pitched', 'fip', 'babip']
    feature_names.extend([f'home_pitcher_{field}' for field in pitcher_fields])
    feature_names.extend([f'away_pitcher_{field}' for field in pitcher_fields])
    
    # Player aggregate fields
    player_fields = ['avg_exit_velocity', 'max_exit_velocity', 'avg_launch_angle',
                     'barrel_rate', 'hard_hit_rate', 'xba', 'xslg', 'xwoba']
    feature_names.extend([f'home_players_{field}' for field in player_fields])
    feature_names.extend([f'away_players_{field}' for field in player_fields])
    
    return feature_names


def get_target_names() -> List[str]:
    """
    Get list of target variable names for ML model training
    
    Returns:
        List of target variable names
    """
    return ['home_won', 'over_hit', 'home_covered', 'home_score', 'away_score', 'total_runs']


if __name__ == "__main__":
    # Example usage and validation
    print("MLB Game Features Schema")
    print("=" * 40)
    
    # Show feature names
    features = get_feature_names()
    targets = get_target_names()
    
    print(f"Total features: {len(features)}")
    print(f"Target variables: {len(targets)}")
    print("\nSample feature names:")
    for i, feature in enumerate(features[:10]):
        print(f"  {i+1}. {feature}")
    print("  ...")
    
    print(f"\nTarget variables:")
    for target in targets:
        print(f"  • {target}")
