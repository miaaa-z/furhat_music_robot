import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# ---- Load data (same as main file) ----
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

song_split_dict = {row['title']: row['split'] for _, row in metadata.iterrows()}
features['split'] = features['song_name'].map(song_split_dict)


def merge_intensity(intensity):
    if intensity <= 1:
        return 0
    elif intensity <= 3:
        return 1
    else:
        return 2


train_data = features[features['split'] == 'Train'].copy()
val_data = features[features['split'] == 'Validation'].copy()

for df in [train_data, val_data]:
    df['intensity_merged'] = df['intensity'].apply(merge_intensity)

X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
y_train = train_data['intensity_merged'].values
y_val = val_data['intensity_merged'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---- Random Search ----
param_dist = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':     ['sqrt', 'log2', None],
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

best_model = search.best_estimator_
val_acc = accuracy_score(y_val, best_model.predict(X_val))
print(f"Validation accuracy with best params: {val_acc:.4f}")