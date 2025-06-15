# # import os
# # import pandas as pd
# # from sklearn.model_selection import train_test_split, cross_val_score
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, classification_report
# # import matplotlib.pyplot as plt

# # # ... other imports ...

# # # === ENHANCED PATH HANDLING ===
# # def get_data_dir():
# #     """Determine the correct data directory with fallback"""
# #     # 1. Check environment variable
# #     env_dir = os.getenv("NBA_DATA_DIR")
# #     if env_dir and os.path.exists(env_dir):
# #         return env_dir
    
# #     # 2. Check Docker environment
# #     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
# #         return "/app/data_files"
    
# #     # 3. Local development paths (add fallbacks if needed)
# #     local_paths = [
# #         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing",
# #         r"C:\Users\Vishal\Desktop\Project\NBA_MASTER\data_files"
# #     ]
    
# #     for path in local_paths:
# #         if os.path.exists(path):
# #             return path
    
# #     # 4. Current directory fallback
# #     return os.getcwd()

# # DATA_DIR = get_data_dir()

# # # === VERBOSE PATH DEBUGGING ===
# # print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# # print(f"Final DATA_DIR: {DATA_DIR}")
# # print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# # if os.path.exists(DATA_DIR):
# #     print(f"Contents: {os.listdir(DATA_DIR)}")
# # print('='*50 + '\n')

# # # === FILE LOADING WITH ENHANCED ERROR HANDLING ===
# # def load_csv_with_retry(filename, retry_paths=[]):
# #     """Load CSV with multiple path attempts"""
# #     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
    
# #     for path in paths_to_try:
# #         try:
# #             print(f"Attempting to load: {path}")
# #             df = pd.read_csv(path)
# #             print(f"✅ Successfully loaded: {filename}")
# #             return df
# #         except FileNotFoundError:
# #             print(f"❌ Not found: {path}")
    
# #     raise FileNotFoundError(f"{filename} not found in any location")

# # # Load data with fallbacks
# # try:
# #     games_df = load_csv_with_retry("Games.csv", [
# #         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\Games.csv"
# #     ])
# #     df = load_csv_with_retry("TeamStatistics.csv", [
# #         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\TeamStatistics.csv"
# #     ])
# # except FileNotFoundError as e:
# #     print(f"⛔ CRITICAL ERROR: {e}")
# #     print("Check your data locations and volume mounts")
# #     exit(1)

# # # === STEP 4: Label Engineering ===
# # games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # # === STEP 5: Data Cleaning ===
# # games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# # games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# # games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# # games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# # numeric_cols_mean = [
# #     'assists', 'blocks', 'steals',
# #     'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted',
# #     'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade',
# #     'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
# #     'foulsPersonal', 'turnovers', 'numMinutes',
# #     'q1Points', 'q2Points', 'q3Points', 'q4Points', 'benchPoints',
# #     'leadChanges', 'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
# #     'pointsSecondChance', 'timesTied', 'seasonWins', 'seasonLosses'
# # ]

# # # === CUSTOM LOGIC: Add score_factor_diff based on assists, field goals, and fouls ===

# # # Step 1: Compute score factor per team in a game
# # df['team_score_factor'] = df['assists'] + df['fieldGoalsMade'] - df['foulsPersonal']

# # # Step 2: Prepare a helper frame with gameId, teamName, and score factor
# # score_helper = df[['gameId', 'teamName', 'team_score_factor']].copy()

# # # Step 3: For each game, reverse team_score_factor to get the opponent's score factor
# # score_helper['opponent_score_factor'] = score_helper.groupby('gameId')['team_score_factor'].transform(lambda x: x[::-1].values)

# # # Step 4: Compute score factor difference
# # score_helper['score_factor_diff'] = score_helper['team_score_factor'] - score_helper['opponent_score_factor']

# # # Step 5: Merge back into df
# # df = df.merge(score_helper[['gameId', 'teamName', 'score_factor_diff']], on=['gameId', 'teamName'], how='left')

# # # === CUSTOM LOGIC: Add score_factor_diff ===
# # # Create a basic score metric using your logic
# # df['team_score_factor'] = df['assists'] + df['fieldGoalsMade'] - df['foulsPersonal']

# # # Compute opponent score factor within the same game
# # # Step 1: Extract relevant columns for processing
# # score_df = df[['gameId', 'teamName', 'team_score_factor']].copy()

# # # Step 2: Reverse score factor within each game (to get opponent's value)
# # score_df['opponent_score_factor'] = score_df.groupby('gameId')['team_score_factor'].transform(lambda x: x[::-1].values)

# # # Step 3: Calculate the difference
# # score_df['score_factor_diff'] = score_df['team_score_factor'] - score_df['opponent_score_factor']

# # # Step 4: Merge back into main dataframe
# # df = df.merge(score_df[['gameId', 'teamName', 'score_factor_diff']], on=['gameId', 'teamName'], how='left')

# # for col in numeric_cols_mean:
# #     df[col] = df[col].fillna(df[col].mean())

# # categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# # for col in categorical_cols_mode:
# #     mode_series = df[col].mode()
# #     df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

# # df['home'] = df['home'].fillna(df['home'].mode()[0])

# # # === STEP 6: Feature Engineering ===
# # df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
# # df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
# # df['shooting_efficiency'] = (
# #     df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# # # === STEP 7: Feature Selection and Merge ===
# # features = [
# #     'home', 'assists', 'blocks', 'steals',
# #     'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted',
# #     'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade',
# #     'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
# #     'foulsPersonal', 'turnovers', 'numMinutes',
# #     'q1Points', 'q2Points', 'q3Points', 'q4Points', 'benchPoints',
# #     'leadChanges', 'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
# #     'pointsSecondChance', 'timesTied', 'seasonWins', 'seasonLosses',
# #     'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency'
# # ]
# # features += ['score_factor_diff']
# # # features += ['score_factor_diff']


# # df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
# #               on='gameId', how='left')
# # features += ['attendance', 'seriesGameNumber', 'seriesGameNumber_missing']

# # df['arenaId'] = df['arenaId'].astype(str)
# # df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # # === STEP 8: Train/Test Split ===
# # X = df[features]
# # y = df['win']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # === STEP 9: Train Model ===
# # model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# # model.fit(X_train, y_train)

# # # === STEP 10: Evaluate Model ===
# # y_pred = model.predict(X_test)
# # print("\n\u2705 Model Evaluation")
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))

# # # === STEP 11: Feature Importance Plot ===
# # importances = model.feature_importances_
# # feature_names = X.columns
# # feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# # plt.figure(figsize=(10, 6))
# # feature_imp.plot(kind='bar')
# # plt.title('Feature Importances')
# # plt.tight_layout()
# # plt.show()

# # # === STEP 12: Cross-Validation ===
# # scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# # print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())

# # # === STEP 13: Predict Single Game ===
# # example_game = pd.DataFrame({
# #     'home': [1], 'assists': [25], 'blocks': [5], 'steals': [8],
# #     'fieldGoalsAttempted': [85], 'fieldGoalsMade': [42], 'threePointersAttempted': [30],
# #     'threePointersMade': [10], 'freeThrowsAttempted': [20], 'freeThrowsMade': [16],
# #     'reboundsDefensive': [30], 'reboundsOffensive': [10], 'reboundsTotal': [40],
# #     'foulsPersonal': [20], 'turnovers': [12], 'numMinutes': [240],
# #     'q1Points': [28], 'q2Points': [26], 'q3Points': [30], 'q4Points': [26],
# #     'benchPoints': [35], 'leadChanges': [5], 'pointsFastBreak': [14],
# #     'pointsFromTurnovers': [18], 'pointsInThePaint': [44],
# #     'pointsSecondChance': [10], 'timesTied': [3], 'seasonWins': [25], 'seasonLosses': [18]
# # })
# # example_game['assist_turnover_ratio'] = example_game['assists'] / (example_game['turnovers'] + 1)
# # example_game['rebound_efficiency'] = example_game['reboundsTotal'] / (example_game['numMinutes'] + 1)
# # example_game['shooting_efficiency'] = (
# #     example_game['fieldGoalsMade'] + example_game['threePointersMade'] + example_game['freeThrowsMade']) / (example_game['fieldGoalsAttempted'] + 1)

# # pred = model.predict(example_game)
# # print("\n\ud83d\udd2e Predicted Outcome:")
# # print("Team will win" if pred[0] == 1 else "Team will lose")


# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # === ENHANCED PATH HANDLING ===
# def get_data_dir():
#     env_dir = os.getenv("NBA_DATA_DIR")
#     if env_dir and os.path.exists(env_dir):
#         return env_dir
#     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
#         return "/app/data_files"
#     local_paths = [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\New_dataset_testing",
#         r"C:\Users\Vishal\Desktop\Project\NBA_MASTER\data_files"
#     ]
#     for path in local_paths:
#         if os.path.exists(path):
#             return path
#     return os.getcwd()

# DATA_DIR = get_data_dir()

# print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# print(f"Final DATA_DIR: {DATA_DIR}")
# print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# if os.path.exists(DATA_DIR):
#     print(f"Contents: {os.listdir(DATA_DIR)}")
# print('='*50 + '\n')

# def load_csv_with_retry(filename, retry_paths=[]):
#     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
#     for path in paths_to_try:
#         try:
#             print(f"Attempting to load: {path}")
#             df = pd.read_csv(path)
#             print(f"✅ Successfully loaded: {filename}")
#             return df
#         except FileNotFoundError:
#             print(f"❌ Not found: {path}")
#     raise FileNotFoundError(f"{filename} not found in any location")

# # Load data
# try:
#     games_df = load_csv_with_retry("Games.csv", [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\Games.csv"])
#     df = load_csv_with_retry("TeamStatistics.csv", [
#         r"C:\Users\Vishal\OneDrive - UNSW\NBA Project Files\NBA_MASTER\Datasets\Python files\TeamStatistics.csv"])
# except FileNotFoundError as e:
#     print(f"⛔ CRITICAL ERROR: {e}")
#     exit(1)

# # Label Engineering
# games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # Data Cleaning
# games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# numeric_cols_mean = [
#     'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
#     'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
#     'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
#     'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
#     'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
#     'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
#     'timesTied', 'seasonWins', 'seasonLosses']

# for col in numeric_cols_mean:
#     df[col] = df[col].fillna(df[col].mean())

# categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# for col in categorical_cols_mode:
#     mode_series = df[col].mode()
#     df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

# df['home'] = df['home'].fillna(df['home'].mode()[0])

# # Feature Engineering
# df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
# df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
# df['shooting_efficiency'] = (
#     df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# df['team_score_factor'] = df['assists'] + df['fieldGoalsMade'] - df['foulsPersonal']
# score_df = df[['gameId', 'teamName', 'team_score_factor']].copy()
# score_df['opponent_score_factor'] = score_df.groupby('gameId')['team_score_factor'].transform(lambda x: x[::-1].values)
# score_df['score_factor_diff'] = score_df['team_score_factor'] - score_df['opponent_score_factor']
# df = df.merge(score_df[['gameId', 'teamName', 'score_factor_diff']], on=['gameId', 'teamName'], how='left')

# # Merge with Games.csv
# df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
#               on='gameId', how='left')
# df['arenaId'] = df['arenaId'].astype(str)
# df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # Feature Selection
# features = numeric_cols_mean + [
#     'home', 'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency',
#     'score_factor_diff', 'attendance', 'seriesGameNumber', 'seriesGameNumber_missing']
# features += [col for col in df.columns if col.startswith('arenaId_')]

# # Model Training
# X = df[features]
# y = df['win']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# model.fit(X_train, y_train)

# # Evaluation
# y_pred = model.predict(X_test)
# print("\n\u2705 Model Evaluation")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Feature Importances
# importances = model.feature_importances_
# feature_names = X.columns
# feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# feature_imp.plot(kind='bar')
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()

# # Confusion Matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# # Cross-Validation
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())

# # Predict Single Game
# example_game = pd.DataFrame({
#     'home': [1], 'assists': [25], 'blocks': [5], 'steals': [8],
#     'fieldGoalsAttempted': [85], 'fieldGoalsMade': [42], 'threePointersAttempted': [30],
#     'threePointersMade': [10], 'freeThrowsAttempted': [20], 'freeThrowsMade': [16],
#     'reboundsDefensive': [30], 'reboundsOffensive': [10], 'reboundsTotal': [40],
#     'foulsPersonal': [20], 'turnovers': [12], 'numMinutes': [240],
#     'q1Points': [28], 'q2Points': [26], 'q3Points': [30], 'q4Points': [26],
#     'benchPoints': [35], 'leadChanges': [5], 'pointsFastBreak': [14],
#     'pointsFromTurnovers': [18], 'pointsInThePaint': [44], 'pointsSecondChance': [10],
#     'timesTied': [3], 'seasonWins': [25], 'seasonLosses': [18]
# })
# example_game['assist_turnover_ratio'] = example_game['assists'] / (example_game['turnovers'] + 1)
# example_game['rebound_efficiency'] = example_game['reboundsTotal'] / (example_game['numMinutes'] + 1)
# example_game['shooting_efficiency'] = (
#     example_game['fieldGoalsMade'] + example_game['threePointersMade'] + example_game['freeThrowsMade']) / (example_game['fieldGoalsAttempted'] + 1)
# example_game['score_factor_diff'] = 0  # dummy value

# # Match features
# for col in features:
#     if col not in example_game.columns:
#         example_game[col] = 0
# example_game = example_game[features]

# pred = model.predict(example_game)
# print("\n\ud83d\udd2e Predicted Outcome:")
# print("Team will win" if pred[0] == 1 else "Team will lose")


# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # === ENHANCED PATH HANDLING ===
# def get_data_dir():
#     env_dir = os.getenv("NBA_DATA_DIR")
#     if env_dir and os.path.exists(env_dir):
#         return env_dir
#     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
#         return "/app/data_files"
#     local_paths = [
#         r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\New_dataset_testing",
#         r"C:\\Users\\Vishal\\Desktop\\Project\\NBA_MASTER\\data_files"
#     ]
#     for path in local_paths:
#         if os.path.exists(path):
#             return path
#     return os.getcwd()

# DATA_DIR = get_data_dir()

# print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# print(f"Final DATA_DIR: {DATA_DIR}")
# print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# if os.path.exists(DATA_DIR):
#     print(f"Contents: {os.listdir(DATA_DIR)}")
# print('='*50 + '\n')

# def load_csv_with_retry(filename, retry_paths=[]):
#     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
#     for path in paths_to_try:
#         try:
#             print(f"Attempting to load: {path}")
#             df = pd.read_csv(path)
#             print(f"✅ Successfully loaded: {filename}")
#             return df
#         except FileNotFoundError:
#             print(f"❌ Not found: {path}")
#     raise FileNotFoundError(f"{filename} not found in any location")

# # Load data
# games_df = load_csv_with_retry("Games.csv")
# df = load_csv_with_retry("TeamStatistics.csv")

# # Label Engineering
# games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # Data Cleaning
# games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# numeric_cols_mean = [
#     'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
#     'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
#     'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
#     'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
#     'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
#     'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
#     'timesTied', 'seasonWins', 'seasonLosses']

# for col in numeric_cols_mean:
#     df[col] = df[col].fillna(df[col].mean())

# categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# for col in categorical_cols_mode:
#     mode_series = df[col].mode()
#     df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

# df['home'] = df['home'].fillna(df['home'].mode()[0])

# # Feature Engineering
# df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
# df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
# df['shooting_efficiency'] = (
#     df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# # Corrected Possession Calculation
# df['Possessions'] = 0.5 * (
#     (df['fieldGoalsAttempted'] + 
#      0.4 * df['freeThrowsAttempted'] - 
#      1.07 * (df['reboundsOffensive'] / (df['reboundsOffensive'] + 1)) + 
#      df['turnovers'])
# )

# # Score factor logic replaced with better feature
# # Example: Net Efficiency Proxy
# df['net_efficiency_proxy'] = df['shooting_efficiency'] + df['assist_turnover_ratio'] - df['turnovers'] / (df['Possessions'] + 1)

# # Merge with Games.csv
# df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
#               on='gameId', how='left')
# df['arenaId'] = df['arenaId'].astype(str)
# df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # Feature Selection
# features = numeric_cols_mean + [
#     'home', 'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency',
#     'Possessions', 'net_efficiency_proxy', 'attendance', 'seriesGameNumber', 'seriesGameNumber_missing']
# features += [col for col in df.columns if col.startswith('arenaId_')]

# # Model Training
# X = df[features]
# y = df['win']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# model.fit(X_train, y_train)

# # Evaluation
# y_pred = model.predict(X_test)
# print("\n\u2705 Model Evaluation")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Feature Importances
# importances = model.feature_importances_
# feature_names = X.columns
# feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# feature_imp.plot(kind='bar')
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()

# # Confusion Matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# # Cross-Validation
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())
# # Predict Single GameI

# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # === ENHANCED PATH HANDLING ===
# def get_data_dir():
#     env_dir = os.getenv("NBA_DATA_DIR")
#     if env_dir and os.path.exists(env_dir):
#         return env_dir
#     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
#         return "/app/data_files"
#     local_paths = [
#         r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\New_dataset_testing",
#         r"C:\\Users\\Vishal\\Desktop\\Project\\NBA_MASTER\\data_files"
#     ]
#     for path in local_paths:
#         if os.path.exists(path):
#             return path
#     return os.getcwd()

# DATA_DIR = get_data_dir()

# print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# print(f"Final DATA_DIR: {DATA_DIR}")
# print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# if os.path.exists(DATA_DIR):
#     print(f"Contents: {os.listdir(DATA_DIR)}")
# print('='*50 + '\n')

# def load_csv_with_retry(filename, retry_paths=[]):
#     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
#     for path in paths_to_try:
#         try:
#             print(f"Attempting to load: {path}")
#             df = pd.read_csv(path)
#             print(f"✅ Successfully loaded: {filename}")
#             return df
#         except FileNotFoundError:
#             print(f"❌ Not found: {path}")
#     raise FileNotFoundError(f"{filename} not found in any location")

# # Load data
# games_df = load_csv_with_retry("Games.csv")
# df = load_csv_with_retry("TeamStatistics.csv")

# # Label Engineering
# games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # Data Cleaning - Games
# games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# # Data Cleaning - Team Stats
# numeric_cols_mean = [
#     'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
#     'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
#     'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
#     'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
#     'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
#     'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
#     'timesTied', 'seasonWins', 'seasonLosses']

# for col in numeric_cols_mean:
#     df[col] = df[col].fillna(df[col].mean())

# categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# for col in categorical_cols_mode:
#     df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

# df['home'] = df['home'].fillna(df['home'].mode()[0])

# # Encode Categorical Columns
# df = pd.get_dummies(df, columns=categorical_cols_mode + ['home'], drop_first=True)

# # Feature Engineering
# df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
# df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
# df['shooting_efficiency'] = (
#     df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# # Possessions
# df['Possessions'] = 0.5 * (
#     (df['fieldGoalsAttempted'] + 
#      0.4 * df['freeThrowsAttempted'] - 
#      1.07 * (df['reboundsOffensive'] / (df['reboundsOffensive'] + 1)) + 
#      df['turnovers'])
# )

# # Net Efficiency Proxy
# df['net_efficiency_proxy'] = df['shooting_efficiency'] + df['assist_turnover_ratio'] - df['turnovers'] / (df['Possessions'] + 1)

# # Merge with Games data
# df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
#               on='gameId', how='left')
# df['arenaId'] = df['arenaId'].astype(str)
# df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # Feature Set
# features = numeric_cols_mean + [
#     'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency',
#     'Possessions', 'net_efficiency_proxy', 'attendance', 'seriesGameNumber', 'seriesGameNumber_missing'
# ]
# # Add all one-hot encoded columns
# encoded_cols = [col for col in df.columns if any(x in col for x in ['arenaId_', 'teamCity_', 'teamName_', 'opponentTeamCity_', 'opponentTeamName_', 'coachId_', 'home_'])]
# features += encoded_cols

# # Model Training
# X = df[features]
# y = df['win']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# model.fit(X_train, y_train)

# # Evaluation
# y_pred = model.predict(X_test)
# print("\n✅ Model Evaluation")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Feature Importances
# importances = model.feature_importances_
# feature_names = X.columns
# feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# feature_imp.plot(kind='bar')
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()

# # Confusion Matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# # Cross-Validation
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print("\n📊 Cross-validation Accuracy:", scores.mean())


# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # === ENHANCED PATH HANDLING ===
# def get_data_dir():
#     env_dir = os.getenv("NBA_DATA_DIR")
#     if env_dir and os.path.exists(env_dir):
#         return env_dir
#     if os.path.exists("/.dockerenv") and os.path.exists("/app/data_files"):
#         return "/app/data_files"
#     local_paths = [
#         r"C:\\Users\\Vishal\\OneDrive - UNSW\\NBA Project Files\\NBA_MASTER\\Datasets\\New_dataset_testing",
#         r"C:\\Users\\Vishal\\Desktop\\Project\\NBA_MASTER\\data_files"
#     ]
#     for path in local_paths:
#         if os.path.exists(path):
#             return path
#     return os.getcwd()

# DATA_DIR = get_data_dir()

# print(f"\n{'='*50}\nDATA DIRECTORY RESOLUTION REPORT")
# print(f"Final DATA_DIR: {DATA_DIR}")
# print(f"Directory exists: {os.path.exists(DATA_DIR)}")
# if os.path.exists(DATA_DIR):
#     print(f"Contents: {os.listdir(DATA_DIR)}")
# print('='*50 + '\n')

# def load_csv_with_retry(filename, retry_paths=[]):
#     paths_to_try = [os.path.join(DATA_DIR, filename)] + retry_paths
#     for path in paths_to_try:
#         try:
#             print(f"Attempting to load: {path}")
#             df = pd.read_csv(path)
#             print(f"✅ Successfully loaded: {filename}")
#             return df
#         except FileNotFoundError:
#             print(f"❌ Not found: {path}")
#     raise FileNotFoundError(f"{filename} not found in any location")

# # Load data
# games_df = load_csv_with_retry("Games.csv")
# df = load_csv_with_retry("TeamStatistics.csv")

# # Label Engineering
# games_df['hometeamwin'] = (games_df['hometeamId'] == games_df['winner']).astype(int)

# # Data Cleaning
# games_df['attendance'] = games_df['attendance'].fillna(games_df['attendance'].mean())
# games_df['seriesGameNumber_missing'] = games_df['seriesGameNumber'].isna().astype(int)
# games_df['seriesGameNumber'] = games_df['seriesGameNumber'].fillna(games_df['seriesGameNumber'].median())
# games_df['arenaId'] = games_df['arenaId'].replace(0, pd.NA).fillna('Unknown')

# numeric_cols_mean = [
#     'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
#     'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted',
#     'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
#     'foulsPersonal', 'turnovers', 'numMinutes', 'q1Points', 'q2Points',
#     'q3Points', 'q4Points', 'benchPoints', 'leadChanges', 'pointsFastBreak',
#     'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance',
#     'timesTied', 'seasonWins', 'seasonLosses']

# for col in numeric_cols_mean:
#     df[col] = df[col].fillna(df[col].mean())

# categorical_cols_mode = ['teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName', 'coachId']
# for col in categorical_cols_mode:
#     mode_series = df[col].mode()
#     df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else "Unknown")

# df['home'] = df['home'].fillna(df['home'].mode()[0])

# # Feature Engineering
# df['assist_turnover_ratio'] = df['assists'] / (df['turnovers'] + 1)
# df['rebound_efficiency'] = df['reboundsTotal'] / (df['numMinutes'] + 1)
# df['shooting_efficiency'] = (
#     df['fieldGoalsMade'] + df['threePointersMade'] + df['freeThrowsMade']) / (df['fieldGoalsAttempted'] + 1)

# df['Possessions'] = 0.5 * (
#     (df['fieldGoalsAttempted'] + 
#      0.4 * df['freeThrowsAttempted'] - 
#      1.07 * (df['reboundsOffensive'] / (df['reboundsOffensive'] + 1)) + 
#      df['turnovers'])
# )

# df['net_efficiency_proxy'] = df['shooting_efficiency'] + df['assist_turnover_ratio'] - df['turnovers'] / (df['Possessions'] + 1)

# # Merge with Games.csv
# df = df.merge(games_df[['gameId', 'attendance', 'arenaId', 'seriesGameNumber', 'seriesGameNumber_missing']], 
#               on='gameId', how='left')
# df['arenaId'] = df['arenaId'].astype(str)
# df = pd.get_dummies(df, columns=['arenaId'], drop_first=True)

# # Feature Selection
# features = numeric_cols_mean + [
#     'home', 'assist_turnover_ratio', 'rebound_efficiency', 'shooting_efficiency',
#     'Possessions', 'net_efficiency_proxy', 'attendance', 'seriesGameNumber', 'seriesGameNumber_missing']
# features += [col for col in df.columns if col.startswith('arenaId_')]

# X = df[features]
# y = df['win']

# # === Feature Scaling ===
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Model Training
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# model.fit(X_train, y_train)

# # Evaluation
# y_pred = model.predict(X_test)
# print("\n✅ Model Evaluation")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Feature Importances
# importances = model.feature_importances_
# feature_names = X.columns
# feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# feature_imp.plot(kind='bar')
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()

# # Confusion Matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()

# # Cross-Validation
# scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
# print("\n📊 Cross-validation Accuracy:", scores.mean())


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

# Evaluation
y_pred = model.predict(X_test)
print("\n\u2705 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importances
importances = model.feature_importances_
feature_names = X.columns
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_imp.plot(kind='bar')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\n\ud83d\udcca Cross-validation Accuracy:", scores.mean())
