import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Facial Expression Classifier (SVM) \n")

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
    print(f" {label:30s}  {count:3d}  ({count/total*100:4.1f}%)")

# Step 4: Prepare features (X) and labels (y)
X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
X_test = test_data[['tempo', 'energy', 'brightness']].values

y_train = train_data['facial_expression'].values
y_val = val_data['facial_expression'].values
y_test = test_data['facial_expression'].values


# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 6: Train SVM with balanced class weights
svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    random_state=10
)
svm.fit(X_train, y_train)

# Step 7: Evaluate on validation set
print("Validation Results")
y_val_pred = svm.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:")
cm_val = confusion_matrix(y_val, y_val_pred)
print(cm_val)

# Step 8: Evaluate on test set
print("Test Results")
y_test_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

# Summary
print("Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
