

import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
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

print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
print(f"Final DATA_DIR: {DATA_DIR}")
print(f"Directory exists: {os.path.exists(DATA_DIR)}")
if os.path.exists(DATA_DIR):
    print(f"Contents: {os.listdir(DATA_DIR)}")
print('='*50 + '\n')

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

# Load data
games_df = load_csv_with_retry("Games.csv")
df = load_csv_with_retry("TeamStatistics.csv")

# Label Engineering
games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# Data Cleaning
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
    'timesTied', 'seasonWins', 'seasonLosses']

for col in numeric_cols_mean:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
for col in categorical_cols_mode:
    mode_series = df[col].mode()
    df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

df['home'] = df['home'].fillna(df['home'].mode()[0])

# Feature Engineering
df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
df['shooting_efficiency'] = (
    df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

df['Possessions'] = 0.5 * (
    (df['fieldGoalsAttempted'] + 
     0.4 * df['freeThrowsAttempted'] - 
     1.07 * (df['reboundsOffensive'] / (df['reboundsOffensive'] + 1)) + 
     df['turnovers'])
)

df['net_efficiency_proxy'] = df['shooting_efficiency'] + df['assist_turnover_ratio'] - df['turnovers'] / (df['Possessions'] + 1)

# Merge with Games.csv
df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
              on='gameId', how='left')
df['arenaId'] = df['arenaId'].astype(str)
df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# Remove duplicates
df = df.drop_duplicates()

# === Outlier Handling using IQR (capping) ===
def cap_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

outlier_cols = numeric_cols_mean + [
    'assist_turnover_ratio',
    'rebound_efficiency',
    'shooting_efficiency',
    'Possessions',
    'net_efficiency_proxy'
]

df = cap_outliers_iqr(df, outlier_cols)

# === Feature Scaling ===
scaler = StandardScaler()
scaled_features = numeric_cols_mean + [
    'assist_turnover_ratio',
    'rebound_efficiency',
    'shooting_efficiency',
    'Possessions',
    'net_efficiency_proxy',
    'attendance',
    'seriesGameNumber'
]

df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Feature Selection
features = numeric_cols_mean + [
    'home', 'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency',
    'Possessions', 'net_efficiency_proxy', 'attendance', 'seriesGameNumber', 'seriesGameNumber_missing']
features += [col for col in df.columns if col.startswith('arenaId_')]

# Model Training
X = df[features]
y = df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# print("Label Balance:")
# print("Full set:\n", y.value_counts(normalize=True))
# print("Train set:\n", y_train.value_counts(normalize=True))
# print("Test set:\n", y_test.value_counts(normalize=True))

# Evaluation
y_pred = model.predict(X_test)
# print("\n\u2705 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importances
importances = model.feature_importances_
feature_names = X.columns
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
feature_imp.plot(kind='bar')
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())


correlations = df[features + ['win']].corr()['win'].sort_values(ascending=False)
print(correlations)
