import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Head Movement Classifier (Random Forest) \n")

# Step 1-2: Load and prepare data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features_v3.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

# Clean whitespace in head_movement column
features['head_movement'] = features['head_movement'].str.strip()

print(f"Total segments: {len(features)}")
print(f"Total songs: {len(metadata)}")

# Create song name to split mapping
song_split_dict = {}
for index, row in metadata.iterrows():
    song_split_dict[row['title']] = row['split']

features['split'] = features['song_name'].map(song_split_dict)

# Step 3: Split into Train, Validation, Test
train_data = features[features['split'] == 'Train']
val_data = features[features['split'] == 'Validation']
test_data = features[features['split'] == 'Test']

print(f"Train:      {len(train_data)} segments")
print(f"Validation: {len(val_data)} segments")
print(f"Test:       {len(test_data)} segments")

# Check head_movement distribution
print("\nHead Movement Distribution (Training set):")
unique, counts = np.unique(train_data['head_movement'], return_counts=True)
total = len(train_data)
for label, count in zip(unique, counts):
    print(f" {label:10s}:  {count:3d}   ({count/total*100:.1f}%)")

# Step 4: Prepare features (X) and labels (y)
label_cols = ['song_name', 'start_time', 'end_time', 'duration',
              'head_movement', 'facial_expression', 'intensity', 'split']
feature_cols = [c for c in train_data.columns if c not in label_cols]

X_train = train_data[feature_cols].values
X_val = val_data[feature_cols].values
X_test = test_data[feature_cols].values

y_train = train_data['head_movement'].values
y_val = val_data['head_movement'].values
y_test = test_data['head_movement'].values

# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Baseline: DummyClassifier
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_val_acc = accuracy_score(y_val, dummy.predict(X_val))
dummy_test_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"Dummy Classifier (most_frequent) - Val Acc: {dummy_val_acc:.4f}, Test Acc: {dummy_test_acc:.4f}")

# Step 6: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Print feature importance
print("Feature Importance:")
feature_names = feature_cols
importances = rf.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

y_train_pred = rf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Step 7: Evaluate on validation set
print("Validation Results")
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:")
cm_val = confusion_matrix(y_val, y_val_pred)
print(cm_val)

# Step 8: Evaluate on test set
print("Test Results")
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

# Sample predictions with probabilities
print("Sample Predictions (first 5 validation samples):")
val_probs = rf.predict_proba(X_val)[:5]
class_names = rf.classes_
for i, (true_label, pred_label, probs) in enumerate(zip(y_val[:5], y_val_pred[:5], val_probs)):
    print(f"\nSample {i+1}:")
    print(f"  True: {true_label}, Predicted: {pred_label}")
    print(f"  Probabilities:")
    for class_name, prob in zip(class_names, probs):
        if prob > 0.01:  # Only show probabilities > 1%
            print(f"    {class_name:15s}: {prob:.3f}")

# Summary
print("Final Results:")
print(f"  Train Accuracy:      {train_acc:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")

# save model
# only when SAVE_MODEL = True, the classifier will be saved
SAVE_MODEL = False

if SAVE_MODEL:
    import joblib, os
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf,     'models/head_classifier.pkl')
    joblib.dump(scaler, 'models/head_scaler.pkl')
    print("Model saved!")
