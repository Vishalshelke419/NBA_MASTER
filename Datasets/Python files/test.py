


import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import os
import pandas as pd
# ... other imports ...

# === ENHANCED PATH HANDLING ===
def get_data_dir():
    """Determine the correct data directory with fallback"""
    # 1. Check environment variable
    env_dir = os.getenv("NBA_DATA_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    
    # 2. Check Docker environment
    if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
        return "/app/data_files"
    
    # 3. Local development paths (add fallbacks if needed)
    local_paths = [
        r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing",
        r"C:\Users\Vishal\Desktop\Project\NBA_MASTER\data_files"
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            return path
    
    # 4. Current directory fallback
    return os.getcwd()

DATA_DIR = get_data_dir()

# === VERBOSE PATH DEBUGGING ===
print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
print(f"Final DATA_DIR: {DATA_DIR}")
print(f"Directory exists: {os.path.exists(DATA_DIR)}")
if os.path.exists(DATA_DIR):
    print(f"Contents: {os.listdir(DATA_DIR)}")
print('='*50 + '\n')

# === FILE LOADING WITH ENHANCED ERROR HANDLING ===
def load_csv_with_retry(filename, retry_paths=[]):
    """Load CSV with multiple path attempts"""
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

# Load data with fallbacks
try:
    games_df = load_csv_with_retry("Games.csv", [
        r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\Games.csv"
    ])
    df = load_csv_with_retry("TeamStatistics.csv", [
        r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\TeamStatistics.csv"
    ])
except FileNotFoundError as e:
    print(f"⛔ CRITICAL ERROR: {e}")
    print("Check your data locations and volume mounts")
    exit(1)

# ... rest of your existing code ...
# === STEP 4: Label Engineering ===
games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# === STEP 5: Data Cleaning ===
games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

numeric_cols_mean = [
    'assists', 'blocks', 'steals',
    'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted',
    'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade',
    'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
    'foulsPersonal', 'turnovers', 'numMinutes',
    'q1Points', 'q2Points', 'q3Points', 'q4Points', 'benchPoints',
    'leadChanges', 'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
    'pointsSecondChance', 'timesTied', 'seasonWins', 'seasonLosses'
]

for col in numeric_cols_mean:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
for col in categorical_cols_mode:
    mode_series = df[col].mode()
    df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

df['home'] = df['home'].fillna(df['home'].mode()[0])

# === STEP 6: Feature Engineering ===
df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
df['shooting_efficiency'] = (
    df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# === STEP 7: Feature Selection and Merge ===
features = [
    'home', 'assists', 'blocks', 'steals',
    'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted',
    'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade',
    'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
    'foulsPersonal', 'turnovers', 'numMinutes',
    'q1Points', 'q2Points', 'q3Points', 'q4Points', 'benchPoints',
    'leadChanges', 'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
    'pointsSecondChance', 'timesTied', 'seasonWins', 'seasonLosses',
    'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency'
]

df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
              on='gameId', how='left')
features += ['attendance', 'seriesGameNumber', 'seriesGameNumber_missing']

df['arenaId'] = df['arenaId'].astype(str)
df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# === STEP 8: Train/Test Split ===
X = df[features]
y = df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 9: Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# === STEP 10: Evaluate Model ===
y_pred = model.predict(X_test)
print("\n\u2705 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === STEP 11: Feature Importance Plot ===
importances = model.feature_importances_
feature_names = X.columns
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_imp.plot(kind='bar')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# === STEP 12: Cross-Validation ===
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())

# === STEP 13: Predict Single Game ===
example_game = pd.DataFrame({
    'home': [1], 'assists': [25], 'blocks': [5], 'steals': [8],
    'fieldGoalsAttempted': [85], 'fieldGoalsMade': [42], 'threePointersAttempted': [30],
    'threePointersMade': [10], 'freeThrowsAttempted': [20], 'freeThrowsMade': [16],
    'reboundsDefensive': [30], 'reboundsOffensive': [10], 'reboundsTotal': [40],
    'foulsPersonal': [20], 'turnovers': [12], 'numMinutes': [240],
    'q1Points': [28], 'q2Points': [26], 'q3Points': [30], 'q4Points': [26],
    'benchPoints': [35], 'leadChanges': [5], 'pointsFastBreak': [14],
    'pointsFromTurnovers': [18], 'pointsInThePaint': [44],
    'pointsSecondChance': [10], 'timesTied': [3], 'seasonWins': [25], 'seasonLosses': [18]
})
example_game['assist_turnover_ratio'] = example_game['assists'] / (example_game['turnovers'] + 1)
example_game['rebound_efficiency'] = example_game['reboundsTotal'] / (example_game['numMinutes'] + 1)
example_game['shooting_efficiency'] = (
    example_game['fieldGoalsMade'] + example_game['threePointersMade'] + example_game['freeThrowsMade']) / (example_game['fieldGoalsAttempted'] + 1)

pred = model.predict(example_game)
print("\n\ud83d\udd2e Predicted Outcome:")
print("Team will win" if pred[0] == 1 else "Team will lose")
