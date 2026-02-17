import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

features['head_movement'] = features['head_movement'].str.strip()
song_split_dict = {row['title']: row['split'] for _, row in metadata.iterrows()}
features['split'] = features['song_name'].map(song_split_dict)

train_data = features[features['split'] == 'Train'].copy()
val_data = features[features['split'] == 'Validation'].copy()

X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
y_train = train_data['head_movement'].values
y_val = val_data['head_movement'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---- Dummy Classifier baseline ----
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
print(f"Dummy Classifier Val Acc: {accuracy_score(y_val, dummy.predict(X_val)):.4f}")

# ---- Random Search ----
param_dist = {
    'n_estimators':      [50, 100, 200, 300],
    'max_depth':         [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', None],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.4f}")
print(f"Validation accuracy with best params: {accuracy_score(y_val, search.best_estimator_.predict(X_val)):.4f}")
