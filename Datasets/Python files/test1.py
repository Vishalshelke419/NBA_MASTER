





# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns

# # === ENHANCED PATH HANDLING ===
# def get_data_dir():
#     """Determine the correct data directory with fallback"""
#     # 1. Check environment variable
#     env_dir = os.getenv("NBA_DATA_DIR")
#     if env_dir and os.path.exists(env_dir):
#         return env_dir
    
#     # 2. Check Docker environment
#     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
#         return "/app/data_files"
    
#     # 3. Local development paths
#     local_paths = [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing",
#         r"C:\Users\Vishal\Desktop\Project\NBA_MASTER\data_files"
#     ]
    
#     for path in local_paths:
#         if os.path.exists(path):
#             return path
    
#     # 4. Current directory fallback
#     return os.getcwd()

# DATA_DIR = get_data_dir()

# # === VERBOSE PATH DEBUGGING ===
# print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# print(f"Final DATA_DIR: {DATA_DIR}")
# print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# if os.path.exists(DATA_DIR):
#     print(f"Contents: {os.listdir(DATA_DIR)}")
# print('='*50 + '\n')

# # === FILE LOADING WITH ENHANCED ERROR HANDLING ===
# def load_csv_with_retry(filename, retry_paths=[]):
#     """Load CSV with multiple path attempts"""
#     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
    
#     for path in paths_to_try:
#         try:
#             print(f"Attempting to load: {path}")
#             df = pd.read_csv(path, low_memory=False)
#             print(f"✅ Successfully loaded: {filename}")
#             return df
#         except FileNotFoundError:
#             print(f"❌ Not found: {path}")
    
#     raise FileNotFoundError(f"{filename} not found in any location")

# # Load data with fallbacks
# try:
#     games_df = load_csv_with_retry("Games.csv", [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\Games.csv"
#     ])
#     team_stats_df = load_csv_with_retry("TeamStatistics.csv", [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\TeamStatistics.csv"
#     ])
# except FileNotFoundError as e:
#     print(f"⛔ CRITICAL ERROR: {e}")
#     print("Check your data locations and volume mounts")
#     exit(1)

# # === STEP 4: Label Engineering ===
# games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # === STEP 5: Data Cleaning ===
# # Fill missing values for Games.csv
# games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# # Define numeric columns to fill
# numeric_cols_mean = [
#     'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
#     'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
#     'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
#     'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
#     'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
#     'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
#     'timesTied', 'seasonWins', 'seasonLosses'
# ]

# # Fill missing values for TeamStatistics.csv
# for col in numeric_cols_mean:
#     team_stats_df[col] = team_stats_df[col].fillna(team_stats_df[col].mean())

# categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# for col in categorical_cols_mode:
#     mode_series = team_stats_df[col].mode()
#     team_stats_df[col] = team_stats_df[col].fillna(
#         mode_series[0] if not mode_series.empty else "Unknown"
#     )

# team_stats_df['home'] = team_stats_df['home'].fillna(team_stats_df['home'].mode()[0])

# # === FIXED OPPONENT SCORE CALCULATION ===
# # Create a temporary DF with game scores
# game_scores = team_stats_df[['gameId', 'teamName', 'teamScore']].copy()

# # Rename columns for merging
# game_scores = game_scores.rename(columns={'teamName': 'opponentTeamName', 'teamScore': 'opponentScore'})

# # Merge to get opponent scores
# team_stats_df = team_stats_df.merge(
#     game_scores, 
#     on=['gameId', 'opponentTeamName'],
#     how='inner',
#     suffixes=('', '_opponent')
# )

# # === STEP 6: ADVANCED FEATURE ENGINEERING ===
# # 1. Calculate advanced efficiency metrics (with safeguards)
# team_stats_df['eFG'] = np.where(
#     team_stats_df['fieldGoalsAttempted'] > 0,
#     (team_stats_df['fieldGoalsMade'] + 0.5 * team_stats_df['threePointersMade']) / team_stats_df['fieldGoalsAttempted'],
#     0
# )
# team_stats_df['TSA'] = team_stats_df['fieldGoalsAttempted'] + 0.44 * team_stats_df['freeThrowsAttempted']
# team_stats_df['TS'] = np.where(
#     team_stats_df['TSA'] > 0,
#     team_stats_df['teamScore'] / (2 * team_stats_df['TSA']),
#     0
# )
# team_stats_df['assist_turnover_ratio'] = team_stats_df['assists'] / (team_stats_df['turnovers'] + 1)
# team_stats_df['off_reb_pct'] = np.where(
#     (team_stats_df['reboundsOffensive'] + team_stats_df['reboundsDefensive']) > 0,
#     team_stats_df['reboundsOffensive'] / (team_stats_df['reboundsOffensive'] + team_stats_df['reboundsDefensive']),
#     0
# )
# team_stats_df['win_prob'] = np.where(
#     (team_stats_df['seasonWins'] + team_stats_df['seasonLosses']) > 0,
#     team_stats_df['seasonWins'] / (team_stats_df['seasonWins'] + team_stats_df['seasonLosses']),
#     0.5
# )

# # 2. Create momentum and clutch features
# team_stats_df['momentum'] = np.where(
#     (team_stats_df['q1Points'] + team_stats_df['q2Points']) > 0,
#     (team_stats_df['q3Points'] + team_stats_df['q4Points']) / (team_stats_df['q1Points'] + team_stats_df['q2Points']),
#     1.0
# )
# team_stats_df['clutch_efficiency'] = np.where(
#     team_stats_df['numMinutes'] > 0,
#     team_stats_df['q4Points'] / (team_stats_df['numMinutes'] / 4),
#     0
# )

# # 3. Create a helper function to safely merge opponent stats
# def get_opponent_stats(df):
#     """Safely merge opponent stats using gameId and opponentTeamName"""
#     # Create a copy of relevant columns
#     opp_stats = df[['gameId', 'teamName', 'eFG', 'TS', 'reboundsTotal', 'turnovers', 'pointsInThePaint']].copy()
    
#     # Rename columns for opponent
#     opp_stats = opp_stats.rename(columns={
#         'teamName': 'opponentTeamName',
#         'eFG': 'opp_eFG',
#         'TS': 'opp_TS',
#         'reboundsTotal': 'opp_reboundsTotal',
#         'turnovers': 'opp_turnovers',
#         'pointsInThePaint': 'opp_paintPoints'
#     })
    
#     # Merge back into main dataframe
#     return df.merge(opp_stats, on=['gameId', 'opponentTeamName'], how='left')

# # Apply the function to merge opponent stats
# team_stats_df = get_opponent_stats(team_stats_df)

# # 4. Create differential features (team vs opponent)
# team_stats_df['eFG_diff'] = team_stats_df['eFG'] - team_stats_df['opp_eFG']
# team_stats_df['rebound_diff'] = team_stats_df['reboundsTotal'] - team_stats_df['opp_reboundsTotal']
# team_stats_df['to_ratio'] = team_stats_df['opp_turnovers'] / (team_stats_df['turnovers'] + 1)  # Higher = opponent has more TOs
# team_stats_df['paint_diff'] = team_stats_df['pointsInThePaint'] - team_stats_df['opp_paintPoints']

# # 5. Calculate game pace and efficiency metrics
# team_stats_df['possessions'] = 0.96 * (
#     team_stats_df['fieldGoalsAttempted'] + 
#     team_stats_df['turnovers'] + 
#     0.44 * team_stats_df['freeThrowsAttempted'] - 
#     team_stats_df['reboundsOffensive']
# )
# team_stats_df['off_rating'] = np.where(
#     team_stats_df['possessions'] > 0,
#     team_stats_df['teamScore'] / team_stats_df['possessions'] * 100,
#     0
# )
# team_stats_df['def_rating'] = np.where(
#     team_stats_df['possessions'] > 0,
#     team_stats_df['opponentScore'] / team_stats_df['possessions'] * 100,
#     0
# )
# team_stats_df['net_rating'] = team_stats_df['off_rating'] - team_stats_df['def_rating']

# # === STEP 7: MERGE WITH GAMES DATA AND FEATURE SELECTION ===
# df = team_stats_df.merge(
#     games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
#     on='gameId', 
#     how='left'
# )

# # Convert arenaId to string and create dummy variables
# df['arenaId'] = df['arenaId'].astype(str)
# df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # Select the most predictive features based on NBA analytics
# features = [
#     'home', 
#     'eFG_diff',          # Shooting efficiency differential
#     'TS',                # True shooting percentage
#     'rebound_diff',      # Rebound differential
#     'to_ratio',          # Turnover ratio
#     'pointsInThePaint',  # High-percentage shots
#     'pointsFromTurnovers', 
#     'momentum',          # Momentum through the game
#     'clutch_efficiency', # Performance in clutch time
#     'attendance',        # Home-court advantage proxy
#     'net_rating',        # Overall team efficiency
#     'off_reb_pct',       # Offensive rebound percentage
#     'assist_turnover_ratio',
#     'seriesGameNumber'   # Playoff context
# ]

# # Add arena dummy features
# features += [col for col in df.columns if col.startswith('arenaId_')]

# # === DATA CLEANING: FINAL CHECKS ===
# # Replace infinite values with NaN
# df.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Fill any remaining missing values with column means
# for col in features:
#     if col in df.columns and df[col].dtype.kind in 'iufc':  # Numeric columns
#         mean_val = df[col].mean()
#         df[col].fillna(mean_val, inplace=True)

# # === STEP 8: HANDLE CLASS IMBALANCE ===
# X = df[features]
# y = df['win']

# # Apply SMOTE to balance classes
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)

# print(f"\n📊 Class Distribution After SMOTE:")
# print(pd.Series(y_res).value_counts())

# # === STEP 9: TRAIN/TEST SPLIT ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X_res, y_res, test_size=0.2, random_state=42
# )

# # === STEP 10: MODEL TRAINING WITH HYPERPARAMETER TUNING ===
# param_grid = {
#     'n_estimators': [100, 150],
#     'max_depth': [8, 10],
#     'min_samples_split': [5, 10],
#     'class_weight': ['balanced']
# }

# model = GridSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

# print("\n🏀 Training model with hyperparameter tuning...")
# model.fit(X_train, y_train)
# print("✅ Model training completed!")

# # Get best model
# best_model = model.best_estimator_
# print(f"\n🌟 Best Model Parameters: {model.best_params_}")

# # === STEP 11: MODEL EVALUATION ===
# y_pred = best_model.predict(X_test)
# y_proba = best_model.predict_proba(X_test)[:, 1]

# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_proba)
# class_report = classification_report(y_test, y_pred)

# print("\n" + "="*50)
# print("📊 MODEL EVALUATION RESULTS")
# print("="*50)
# print(f"✅ Accuracy: {accuracy:.4f}")
# print(f"📈 ROC AUC: {roc_auc:.4f}")
# print("\n📝 Classification Report:")
# print(class_report)

# # Cross-validation scores
# cv_scores = cross_val_score(best_model, X_res, y_res, cv=5, scoring='accuracy')
# print("\n🔍 Cross-validation Accuracy Scores:")
# print([f"{score:.4f}" for score in cv_scores])
# print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# # Feature Importances
# importances = best_model.feature_importances_
# feature_names = X.columns
# feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# plt.figure(figsize=(12, 8))
# feature_imp.nlargest(15).plot(kind='barh')
# plt.title('Top 15 Feature Importances')
# plt.tight_layout()
# plt.savefig('feature_importances.png')
# plt.show()

# # Confusion Matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Loss', 'Win'])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix")
# plt.savefig('confusion_matrix.png')
# plt.show()

# # === STEP 12: PREDICT SINGLE GAME ===
# # Create example game with realistic values
# example_game = pd.DataFrame({
#     'home': [1],
#     'eFG_diff': [0.05],       # Team has 5% better effective FG%
#     'TS': [0.58],             # 58% true shooting
#     'rebound_diff': [5],      # +5 rebound differential
#     'to_ratio': [1.2],        # Opponent has 20% more turnovers
#     'pointsInThePaint': [48],
#     'pointsFromTurnovers': [18],
#     'momentum': [1.1],        # Better performance in second half
#     'clutch_efficiency': [1.8], # Strong 4th quarter performance
#     'attendance': [18000],
#     'net_rating': [5.2],      # +5.2 net rating per 100 possessions
#     'off_reb_pct': [0.28],    # 28% offensive rebound percentage
#     'assist_turnover_ratio': [2.1],
#     'seriesGameNumber': [3]   # Game 3 of playoff series
# })

# # Add arenaId features (set to 0 for simplicity)
# for col in features:
#     if col.startswith('arenaId_'):
#         example_game[col] = 0

# # Ensure all features are present
# example_game = example_game[features]

# # Make prediction
# pred = best_model.predict(example_game)
# pred_proba = best_model.predict_proba(example_game)[0]
# win_prob = pred_proba[1] * 100

# print("\n" + "="*50)
# print("🎯 SINGLE GAME PREDICTION")
# print("="*50)
# print("Predicted Outcome:", "Win" if pred[0] == 1 else "Loss")
# print(f"Prediction Confidence: {win_prob:.1f}%")
# print("\nKey Factors Contributing to Prediction:")
# print("--------------------------------------")

# # Show top contributing factors
# example_importances = pd.Series(
#     best_model.feature_importances_ * example_game.values[0],
#     index=features
# ).sort_values(ascending=False)

# for feature, importance in example_importances.head(5).items():
#     value = example_game[feature].values[0]
#     print(f"{feature}: {value:.2f} (Impact: {importance:.4f})") 



import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === ENHANCED PATH HANDLING ===
def get_data_dir():
    env_dir = os.getenv("NBA_DATA_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
        return "/app/data_files"
    local_paths = [
        r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\New_dataset_testing",
        r"C:\\Users\\Vishal\\Desktop\\Project\\NBA_MASTER\\data_files"
    ]
    for path in local_paths:
        if os.path.exists(path):
            return path
    return os.getcwd()

DATA_DIR = get_data_dir()

def load_csv_with_retry(filename, retry_paths=[]):
    paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
    for path in paths_to_try:
        try:
            print(f"Attempting to load: {path}")
            df = pd.read_csv(path)
            print(f"✅ Successfully loaded: {filename}")
            return df
        except FileNotFoundError:
            print(f"❌ Not found: {path}")
    raise FileNotFoundError(f"{filename} not found in any location")

# === LOAD DATA ===
games_df = load_csv_with_retry("Games.csv", [
    r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\Python files\\Games.csv"])
df = load_csv_with_retry("TeamStatistics.csv", [
    r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\Python files\\TeamStatistics.csv"])

# === DATA CLEANING ===
games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)
games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

numeric_cols_mean = [
    'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
    'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
    'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
    'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
    'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
    'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
    'timesTied', 'seasonWins', 'seasonLosses', 'teamScore']

for col in numeric_cols_mean:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
for col in categorical_cols_mode:
    mode_series = df[col].mode()
    df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")
df['home'] = df['home'].fillna(df['home'].mode()[0])

# === ADVANCED FEATURE ENGINEERING ===
df['eFG%'] = (df['fieldGoalsMade'] + 0.5 * df['threePointersMade']) / df['fieldGoalsAttempted']
df['TS%'] = df['teamScore'] / (2 * (df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted']))
df['PointsInPaintRatio'] = df['pointsInThePaint'] / df['teamScore']
df['Turnover%'] = df['turnovers'] / (df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted'] + df['turnovers'])
df['OffReb%'] = df['reboundsOffensive'] / (df['reboundsOffensive'] + df['reboundsDefensive'])
df['Possessions'] = 0.5 * (
    (df['fieldGoalsAttempted'] + 0.4 * df['freeThrowsAttempted'] - 1.07 *
     (df['reboundsOffensive'] / (df['reboundsOffensive'] + df['reboundsDefensive'])) *
     (df['fieldGoalsMade'] - df['reboundsOffensive']) + df['turnovers'])
)
df['DefensiveImpact'] = df['steals'] + df['blocks']
df['NetRating'] = (df['teamScore'] - df['opponentTeamScore']) / df['Possessions']
df['HomeImpact'] = df['home'].astype(int)

# === FINAL FEATURE SELECTION ===
selected_features = [
    'eFG%', 'TS%', 'PointsInPaintRatio',
    'Turnover%', 'OffReb%',
    'DefensiveImpact', 'NetRating', 'HomeImpact'
]

df = df.dropna(subset=selected_features)
X = df[selected_features]
y = df['teamScore'] > df['opponentTeamScore']  # Binary classification: win or not

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL TRAINING ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUATION ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Loss', 'Win'])
disp.plot()
plt.show()
