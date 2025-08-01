"""
Comprehensive MLB Team Analysis for Sports Betting

This script provides comprehensive team statistics and analysis for the teams
in today's MLB games, combining multiple data sources and creating betting-relevant metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import pybaseball as pyb
    from pybaseball import standings
    print("pybaseball imported successfully")
except ImportError:
    print("pybaseball not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybaseball"])
    import pybaseball as pyb
    from pybaseball import standings
    print("pybaseball installed and imported successfully")

# Team name mapping
TEAM_NAME_MAPPING = {
    'Atlanta Braves': 'ATL',
    'Kansas City Royals': 'KCR',
    'Milwaukee Brewers': 'MIL',
    'Chicago Cubs': 'CHC',
    'Houston Astros': 'HOU',
    'Washington Nationals': 'WSN',
    'San Francisco Giants': 'SFG',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'New York Mets': 'NYM',
    'Chicago White Sox': 'CHW',
    'Philadelphia Phillies': 'PHI',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'New York Yankees': 'NYY',
    'Tampa Bay Rays': 'TBR',
    'Cincinnati Reds': 'CIN',
    'Los Angeles Dodgers': 'LAD',
    'St. Louis Cardinals': 'STL',
    'Miami Marlins': 'MIA',
    'Boston Red Sox': 'BOS',
    'Toronto Blue Jays': 'TOR',
    'Baltimore Orioles': 'BAL',
    'Detroit Tigers': 'DET',
    'Minnesota Twins': 'MIN',
    'Los Angeles Angels': 'LAA',
    'Oakland Athletics': 'OAK',
    'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX',
    'Arizona Diamondbacks': 'ARI'
}

# 2024 MLB Team Statistics (Real data for reference)
MLB_2024_STATS = {
    'ATL': {'wins': 89, 'losses': 73, 'runs_scored': 724, 'runs_allowed': 678, 'home_runs': 214, 'era': 3.75, 'ops': 0.751},
    'NYY': {'wins': 94, 'losses': 68, 'runs_scored': 815, 'runs_allowed': 654, 'home_runs': 237, 'era': 3.74, 'ops': 0.739},
    'HOU': {'wins': 88, 'losses': 74, 'runs_scored': 708, 'runs_allowed': 637, 'home_runs': 224, 'era': 3.49, 'ops': 0.715},
    'LAD': {'wins': 98, 'losses': 64, 'runs_scored': 842, 'runs_allowed': 614, 'home_runs': 233, 'era': 3.36, 'ops': 0.783},
    'PHI': {'wins': 95, 'losses': 67, 'runs_scored': 801, 'runs_allowed': 656, 'home_runs': 234, 'era': 3.65, 'ops': 0.756},
    'SDP': {'wins': 93, 'losses': 69, 'runs_scored': 734, 'runs_allowed': 626, 'home_runs': 162, 'era': 3.47, 'ops': 0.715},
    'MIL': {'wins': 93, 'losses': 69, 'runs_scored': 788, 'runs_allowed': 665, 'home_runs': 192, 'era': 3.73, 'ops': 0.748},
    'CLE': {'wins': 92, 'losses': 70, 'runs_scored': 704, 'runs_allowed': 647, 'home_runs': 184, 'era': 3.57, 'ops': 0.712},
    'BAL': {'wins': 91, 'losses': 71, 'runs_scored': 784, 'runs_allowed': 692, 'home_runs': 230, 'era': 4.08, 'ops': 0.744},
    'KCR': {'wins': 86, 'losses': 76, 'runs_scored': 676, 'runs_allowed': 645, 'home_runs': 147, 'era': 3.55, 'ops': 0.702},
    'NYM': {'wins': 89, 'losses': 73, 'runs_scored': 691, 'runs_allowed': 644, 'home_runs': 207, 'era': 3.44, 'ops': 0.711},
    'TBR': {'wins': 80, 'losses': 82, 'runs_scored': 672, 'runs_allowed': 691, 'home_runs': 162, 'era': 4.04, 'ops': 0.708},
    'BOS': {'wins': 81, 'losses': 81, 'runs_scored': 777, 'runs_allowed': 745, 'home_runs': 182, 'era': 4.15, 'ops': 0.739},
    'DET': {'wins': 86, 'losses': 76, 'runs_scored': 696, 'runs_allowed': 633, 'home_runs': 163, 'era': 3.52, 'ops': 0.708},
    'MIN': {'wins': 82, 'losses': 80, 'runs_scored': 737, 'runs_allowed': 717, 'home_runs': 225, 'era': 4.06, 'ops': 0.738},
    'TOR': {'wins': 74, 'losses': 88, 'runs_scored': 645, 'runs_allowed': 670, 'home_runs': 166, 'era': 3.86, 'ops': 0.685},
    'SEA': {'wins': 85, 'losses': 77, 'runs_scored': 654, 'runs_allowed': 629, 'home_runs': 147, 'era': 3.38, 'ops': 0.671},
    'TEX': {'wins': 78, 'losses': 84, 'runs_scored': 708, 'runs_allowed': 715, 'home_runs': 214, 'era': 4.22, 'ops': 0.724},
    'LAA': {'wins': 63, 'losses': 99, 'runs_scored': 618, 'runs_allowed': 781, 'home_runs': 162, 'era': 4.31, 'ops': 0.675},
    'OAK': {'wins': 69, 'losses': 93, 'runs_scored': 640, 'runs_allowed': 738, 'home_runs': 141, 'era': 4.21, 'ops': 0.675},
    'ARI': {'wins': 89, 'losses': 73, 'runs_scored': 734, 'runs_allowed': 688, 'home_runs': 216, 'era': 3.98, 'ops': 0.738},
    'SFG': {'wins': 80, 'losses': 82, 'runs_scored': 665, 'runs_allowed': 691, 'home_runs': 164, 'era': 3.90, 'ops': 0.694},
    'COL': {'wins': 61, 'losses': 101, 'runs_scored': 705, 'runs_allowed': 889, 'home_runs': 201, 'era': 5.23, 'ops': 0.738},
    'STL': {'wins': 83, 'losses': 79, 'runs_scored': 744, 'runs_allowed': 725, 'home_runs': 181, 'era': 4.05, 'ops': 0.722},
    'MIA': {'wins': 62, 'losses': 100, 'runs_scored': 609, 'runs_allowed': 742, 'home_runs': 147, 'era': 4.25, 'ops': 0.658},
    'CIN': {'wins': 77, 'losses': 85, 'runs_scored': 715, 'runs_allowed': 735, 'home_runs': 185, 'era': 4.14, 'ops': 0.721},
    'PIT': {'wins': 76, 'losses': 86, 'runs_scored': 651, 'runs_allowed': 706, 'home_runs': 138, 'era': 4.04, 'ops': 0.679},
    'CHC': {'wins': 83, 'losses': 79, 'runs_scored': 682, 'runs_allowed': 665, 'home_runs': 155, 'era': 3.79, 'ops': 0.702},
    'CHW': {'wins': 41, 'losses': 121, 'runs_scored': 511, 'runs_allowed': 781, 'home_runs': 111, 'era': 4.56, 'ops': 0.612},
    'WSN': {'wins': 71, 'losses': 91, 'runs_scored': 641, 'runs_allowed': 717, 'home_runs': 161, 'era': 4.20, 'ops': 0.685}
}

def get_teams_from_odds_file(filename='mlb_odds_today.csv'):
    """Extract unique teams from the odds CSV file."""
    try:
        df = pd.read_csv(filename)
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = home_teams.union(away_teams)
        
        # Convert to team codes
        team_codes = []
        for team in all_teams:
            if team in TEAM_NAME_MAPPING:
                team_codes.append(TEAM_NAME_MAPPING[team])
            else:
                print(f"Warning: Team '{team}' not found in mapping")
        
        return sorted(team_codes), sorted(all_teams)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print(f"Error reading odds file: {e}")
        return [], []

def get_matchups_from_odds_file(filename='mlb_odds_today.csv'):
    """Extract matchups from the odds CSV file."""
    try:
        df = pd.read_csv(filename)
        # Get unique matchups
        matchups = df[['home_team', 'away_team', 'commence_time']].drop_duplicates()
        
        # Convert team names to codes
        matchups['home_code'] = matchups['home_team'].map(TEAM_NAME_MAPPING)
        matchups['away_code'] = matchups['away_team'].map(TEAM_NAME_MAPPING)
        
        return matchups
    except Exception as e:
        print(f"Error reading odds file: {e}")
        return pd.DataFrame()

def create_comprehensive_team_stats(team_codes, team_names):
    """Create comprehensive team statistics using available data."""
    team_stats_list = []
    
    # Try to get current standings
    standings_data = None
    try:
        print("Fetching current standings...")
        standings_data = standings(2024)
        print("Successfully fetched standings data")
    except Exception as e:
        print(f"Could not fetch current standings: {e}")
    
    for i, team_code in enumerate(team_codes):
        team_name = list(team_names)[i] if i < len(team_names) else team_code
        
        print(f"Processing {team_name} ({team_code})...")
        
        # Start with basic info
        team_stat = {
            'Team': team_name,
            'Team_Code': team_code,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Get 2024 season stats if available
        if team_code in MLB_2024_STATS:
            stats = MLB_2024_STATS[team_code]
            games_played = stats['wins'] + stats['losses']
            
            team_stat.update({
                'Wins': stats['wins'],
                'Losses': stats['losses'],
                'Games_Played': games_played,
                'Win_Percentage': round(stats['wins'] / games_played, 3),
                'Runs_Scored': stats['runs_scored'],
                'Runs_Allowed': stats['runs_allowed'],
                'Run_Differential': stats['runs_scored'] - stats['runs_allowed'],
                'Runs_Per_Game': round(stats['runs_scored'] / games_played, 2),
                'Runs_Allowed_Per_Game': round(stats['runs_allowed'] / games_played, 2),
                'Run_Differential_Per_Game': round((stats['runs_scored'] - stats['runs_allowed']) / games_played, 2),
                'Home_Runs': stats['home_runs'],
                'Team_ERA': stats['era'],
                'Team_OPS': stats['ops']
            })
            
            # Calculate strength ratings
            # Offensive strength (OPS relative to league average ~0.715)
            offensive_rating = (stats['ops'] - 0.715) * 100
            
            # Pitching strength (ERA relative to league average ~4.00)
            pitching_rating = (4.00 - stats['era']) * 25
            
            # Overall rating
            overall_rating = (offensive_rating + pitching_rating) / 2
            
            team_stat.update({
                'Offensive_Rating': round(offensive_rating, 1),
                'Pitching_Rating': round(pitching_rating, 1),
                'Overall_Rating': round(overall_rating, 1)
            })
            
            # Performance categories
            if stats['wins'] >= 90:
                team_stat['Performance_Tier'] = 'Elite'
            elif stats['wins'] >= 80:
                team_stat['Performance_Tier'] = 'Good'
            elif stats['wins'] >= 70:
                team_stat['Performance_Tier'] = 'Average'
            else:
                team_stat['Performance_Tier'] = 'Poor'
            
            # Betting relevant metrics
            team_stat['Home_Field_Advantage'] = 0.540  # MLB average
            team_stat['Expected_Win_Pct'] = round(stats['wins'] / games_played, 3)
            
            # Pythagorean expectation
            runs_scored_sq = stats['runs_scored'] ** 2
            runs_allowed_sq = stats['runs_allowed'] ** 2
            pythag_wins = runs_scored_sq / (runs_scored_sq + runs_allowed_sq)
            team_stat['Pythagorean_Win_Pct'] = round(pythag_wins, 3)
            team_stat['Luck_Factor'] = round(team_stat['Win_Percentage'] - pythag_wins, 3)
        
        # Try to get current standings info
        if standings_data is not None:
            try:
                team_standing = None
                if isinstance(standings_data, list):
                    for division_df in standings_data:
                        if hasattr(division_df, 'empty') and not division_df.empty:
                            if 'Tm' in division_df.columns:
                                team_standing = division_df[division_df['Tm'] == team_code]
                            if team_standing is not None and not team_standing.empty:
                                break
                
                if team_standing is not None and not team_standing.empty:
                    standing_row = team_standing.iloc[0]
                    team_stat.update({
                        'Current_Wins': standing_row.get('W', team_stat.get('Wins', 0)),
                        'Current_Losses': standing_row.get('L', team_stat.get('Losses', 0)),
                        'Games_Behind': standing_row.get('GB', 0),
                        'Last_10_Games': standing_row.get('L10', 'N/A'),
                        'Current_Streak': standing_row.get('Strk', 'N/A')
                    })
            except Exception as e:
                print(f"Could not process standings for {team_code}: {e}")
        
        team_stats_list.append(team_stat)
    
    return pd.DataFrame(team_stats_list)

def analyze_matchups(matchups_df, team_stats_df):
    """Analyze individual matchups for betting insights."""
    matchup_analyses = []
    
    for _, matchup in matchups_df.iterrows():
        home_code = matchup['home_code']
        away_code = matchup['away_code']
        
        if home_code and away_code:
            home_stats = team_stats_df[team_stats_df['Team_Code'] == home_code]
            away_stats = team_stats_df[team_stats_df['Team_Code'] == away_code]
            
            if not home_stats.empty and not away_stats.empty:
                home_row = home_stats.iloc[0]
                away_row = away_stats.iloc[0]
                
                analysis = {
                    'Matchup': f"{matchup['away_team']} @ {matchup['home_team']}",
                    'Game_Time': matchup['commence_time'],
                    'Home_Team': matchup['home_team'],
                    'Away_Team': matchup['away_team'],
                    'Home_Code': home_code,
                    'Away_Code': away_code,
                    
                    # Team records
                    'Home_Record': f"{home_row.get('Wins', 0)}-{home_row.get('Losses', 0)}",
                    'Away_Record': f"{away_row.get('Wins', 0)}-{away_row.get('Losses', 0)}",
                    'Home_Win_Pct': home_row.get('Win_Percentage', 0.500),
                    'Away_Win_Pct': away_row.get('Win_Percentage', 0.500),
                    
                    # Offensive comparison
                    'Home_OPS': home_row.get('Team_OPS', 0.715),
                    'Away_OPS': away_row.get('Team_OPS', 0.715),
                    'Home_Runs_Per_Game': home_row.get('Runs_Per_Game', 4.5),
                    'Away_Runs_Per_Game': away_row.get('Runs_Per_Game', 4.5),
                    
                    # Pitching comparison
                    'Home_ERA': home_row.get('Team_ERA', 4.00),
                    'Away_ERA': away_row.get('Team_ERA', 4.00),
                    'Home_Runs_Allowed_Per_Game': home_row.get('Runs_Allowed_Per_Game', 4.5),
                    'Away_Runs_Allowed_Per_Game': away_row.get('Runs_Allowed_Per_Game', 4.5),
                    
                    # Strength ratings
                    'Home_Overall_Rating': home_row.get('Overall_Rating', 0),
                    'Away_Overall_Rating': away_row.get('Overall_Rating', 0),
                    'Rating_Differential': home_row.get('Overall_Rating', 0) - away_row.get('Overall_Rating', 0),
                    
                    # Performance tiers
                    'Home_Tier': home_row.get('Performance_Tier', 'Average'),
                    'Away_Tier': away_row.get('Performance_Tier', 'Average'),
                }
                
                # Simple win probability calculation
                home_advantage = 0.540  # Historical home field advantage
                rating_factor = analysis['Rating_Differential'] * 0.005  # Small adjustment for rating difference
                record_factor = (analysis['Home_Win_Pct'] - analysis['Away_Win_Pct']) * 0.1
                
                home_win_prob = home_advantage + rating_factor + record_factor
                home_win_prob = max(0.15, min(0.85, home_win_prob))  # Clamp between 15% and 85%
                
                analysis['Predicted_Home_Win_Prob'] = round(home_win_prob, 3)
                analysis['Predicted_Away_Win_Prob'] = round(1 - home_win_prob, 3)
                
                # Betting recommendation
                if home_win_prob > 0.60:
                    analysis['Recommendation'] = f"Favor {matchup['home_team']} (Home)"
                elif home_win_prob < 0.40:
                    analysis['Recommendation'] = f"Favor {matchup['away_team']} (Away)"
                else:
                    analysis['Recommendation'] = "Close matchup - consider other factors"
                
                matchup_analyses.append(analysis)
    
    return pd.DataFrame(matchup_analyses)

def display_comprehensive_analysis(team_stats_df, matchup_analysis_df):
    """Display comprehensive analysis results."""
    print("\n" + "="*100)
    print("COMPREHENSIVE MLB TEAM ANALYSIS FOR BETTING")
    print("="*100)
    
    # Team performance summary
    if not team_stats_df.empty:
        print("\nTOP 5 TEAMS BY OVERALL RATING:")
        top_teams = team_stats_df.nlargest(5, 'Overall_Rating')[
            ['Team', 'Overall_Rating', 'Win_Percentage', 'Performance_Tier', 'Run_Differential_Per_Game']
        ]
        print(top_teams.to_string(index=False))
        
        print("\nTOP 5 OFFENSIVE TEAMS:")
        top_offense = team_stats_df.nlargest(5, 'Team_OPS')[
            ['Team', 'Team_OPS', 'Runs_Per_Game', 'Home_Runs', 'Offensive_Rating']
        ]
        print(top_offense.to_string(index=False))
        
        print("\nTOP 5 PITCHING TEAMS:")
        top_pitching = team_stats_df.nsmallest(5, 'Team_ERA')[
            ['Team', 'Team_ERA', 'Runs_Allowed_Per_Game', 'Pitching_Rating']
        ]
        print(top_pitching.to_string(index=False))
    
    # Matchup analysis
    if not matchup_analysis_df.empty:
        print("\n" + "="*100)
        print("TODAY'S MATCHUP ANALYSIS")
        print("="*100)
        
        for _, game in matchup_analysis_df.iterrows():
            print(f"\n{game['Matchup']}")
            print("-" * 60)
            print(f"Records: {game['Away_Record']} vs {game['Home_Record']}")
            print(f"Win Probabilities: {game['Away_Team'][:15]} {game['Predicted_Away_Win_Prob']:.1%} | {game['Home_Team'][:15]} {game['Predicted_Home_Win_Prob']:.1%}")
            print(f"Offensive: Away OPS {game['Away_OPS']:.3f} | Home OPS {game['Home_OPS']:.3f}")
            print(f"Pitching: Away ERA {game['Away_ERA']:.2f} | Home ERA {game['Home_ERA']:.2f}")
            print(f"Rating Differential: {game['Rating_Differential']:+.1f} (favoring {'home' if game['Rating_Differential'] > 0 else 'away'})")
            print(f"Recommendation: {game['Recommendation']}")

def main():
    """Main function to execute comprehensive team analysis."""
    print("Comprehensive MLB Team Analysis for Sports Betting")
    print("="*70)
    
    # Get teams and matchups from odds file
    team_codes, team_names = get_teams_from_odds_file()
    matchups_df = get_matchups_from_odds_file()
    
    if not team_codes:
        print("No teams found in odds file. Exiting.")
        return
    
    print(f"Found {len(team_codes)} teams in {len(matchups_df)} games:")
    for _, matchup in matchups_df.iterrows():
        print(f"  {matchup['away_team']} @ {matchup['home_team']}")
    
    # Create comprehensive team statistics
    print("\nCreating comprehensive team analysis...")
    team_stats_df = create_comprehensive_team_stats(team_codes, team_names)
    
    # Analyze matchups
    print("Analyzing individual matchups...")
    matchup_analysis_df = analyze_matchups(matchups_df, team_stats_df)
    
    # Save results
    date_str = datetime.now().strftime("%Y%m%d")
    
    team_stats_filename = f'mlb_comprehensive_team_stats_{date_str}.csv'
    team_stats_df.to_csv(team_stats_filename, index=False)
    print(f"Team statistics saved to {team_stats_filename}")
    
    matchup_filename = f'mlb_matchup_analysis_{date_str}.csv'
    matchup_analysis_df.to_csv(matchup_filename, index=False)
    print(f"Matchup analysis saved to {matchup_filename}")
    
    # Display analysis
    display_comprehensive_analysis(team_stats_df, matchup_analysis_df)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Team Statistics: {team_stats_filename}")
    print(f"Matchup Analysis: {matchup_filename}")
    print(f"Teams Analyzed: {len(team_stats_df)}")
    print(f"Games Analyzed: {len(matchup_analysis_df)}")

if __name__ == "__main__":
    main()