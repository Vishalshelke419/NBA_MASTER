import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# === Step 1: Load Team Dataset ===
data_path = r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing\TeamStatistics.csv"
df = pd.read_csv(data_path)
games_path = r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing\Games.csv"
games_df = pd.read_csv(games_path)


# Create binary win column
games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# Clean relevant columns
games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')


# === Step 2: Handle Missing Values ===
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
    if not mode_series.empty:
        df[col] = df[col].fillna(mode_series[0])
    else:
        df[col] = df[col].fillna("Unknown")

df['home'] = df['home'].fillna(df['home'].mode()[0])

# === Step 3: Feature Engineering ===
df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
df['shooting_efficiency'] = (
    df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# === Step 4: Select Features ===
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
              on='gameId', 
              how='left')
features += ['attendance', 'seriesGameNumber', 'seriesGameNumber_missing']
df['arenaId'] = df['arenaId'].astype(str)
df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)


X = df[features]
y = df['win']

# === Step 5: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# === Step 7: Evaluate Model ===
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Step 8: Feature Importance Plot ===
importances = model.feature_importances_
feature_names = X.columns
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_imp.plot(kind='bar')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# === Step 9: Cross-Validation Accuracy ===
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\n📊 Cross-validation Accuracy:", scores.mean())

# === Step 10: Predict for a new example game ===
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
print("\n🔮 Predicted Outcome:")
print("Team will win" if pred[0] == 1 else "Team will lose")

