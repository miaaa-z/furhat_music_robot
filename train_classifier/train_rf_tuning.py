import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features_v2.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

features['head_movement'] = features['head_movement'].str.strip()
song_split_dict = {row['title']: row['split'] for _, row in metadata.iterrows()}
features['split'] = features['song_name'].map(song_split_dict)

train_data = features[features['split'] == 'Train']
val_data = features[features['split'] == 'Validation']

label_cols = ['song_name', 'start_time', 'end_time', 'duration',
              'head_movement', 'facial_expression', 'intensity', 'split']
feature_cols = [c for c in train_data.columns if c not in label_cols]

X_train = train_data[feature_cols].values
X_val = val_data[feature_cols].values
y_train = train_data['head_movement'].values
y_val = val_data['head_movement'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Random Search
param_dist = {
    'n_estimators':      [50, 100, 200, 300],
    'max_depth':         [3, 5, 8, 10, None],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf':  [3, 5, 8, 10],       
    'max_features':      ['sqrt', 'log2'],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=60,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV accuracy (cross-val): {search.best_score_:.4f}")

best_model = search.best_estimator_

train_acc = accuracy_score(y_train, best_model.predict(X_train))
val_acc = accuracy_score(y_val, best_model.predict(X_val))

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Overfit gap:    {train_acc - val_acc:.4f}")
print("\nClassification Report (Val):")
print(classification_report(y_val, best_model.predict(X_val)))
