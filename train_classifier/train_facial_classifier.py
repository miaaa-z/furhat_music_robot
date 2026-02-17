import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Facial Expression Classifier (Random Forest - Original 21 Classes) \n")

# Step 1-2: Load and prepare data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

# Clean whitespace in facial_expression column
features['facial_expression'] = features['facial_expression'].str.strip()

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

# Check facial_expression distribution
print("\nFacial Expression Distribution (Training set):")
unique, counts = np.unique(train_data['facial_expression'], return_counts=True)
total = len(train_data)
for label, count in zip(unique, counts):
    print(f"  {label:25s}: {count:3d} ({count/total*100:5.1f}%)")

# Step 4: Prepare features (X) and labels (y)
X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
X_test = test_data[['tempo', 'energy', 'brightness']].values

y_train = train_data['facial_expression'].values
y_val = val_data['facial_expression'].values
y_test = test_data['facial_expression'].values

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

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
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete\n")

# Print feature importance
print("Feature Importance:")
feature_names = ['tempo', 'energy', 'brightness']
importances = rf.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

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


# Summary
print("\n")
print("Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
