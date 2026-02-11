import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("Training Intensity Classifier (SVM with Class Weights) \n")

# Step 1: Load data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

print(f"Total segments: {len(features)}")
print(f"Total songs: {len(metadata)}")

# Step 2: Create song name to split mapping
song_split_dict = {}
for index, row in metadata.iterrows():
    song_name = row['title']
    split_type = row['split']
    song_split_dict[song_name] = split_type

features['split'] = features['song_name'].map(song_split_dict)

# Step 3: Split into Train, Validation, Test
train_data = features[features['split'] == 'Train']
val_data = features[features['split'] == 'Validation']
test_data = features[features['split'] == 'Test']

print(f"Train:      {len(train_data)} segments")
print(f"Validation: {len(val_data)} segments")
print(f"Test:       {len(test_data)} segments")

# Step 4: Prepare features (X) and labels (y)
X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
X_test = test_data[['tempo', 'energy', 'brightness']].values

y_train = train_data['intensity'].values
y_val = val_data['intensity'].values
y_test = test_data['intensity'].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# try new improvement: Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}\n")

# Step 6: Train SVM with class weights
svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=class_weight_dict)
svm.fit(X_train, y_train)
print("Training complete\n")

# Step 7: Evaluate on validation set
print("Validation Results")
y_val_pred = svm.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\n")

# Step 8: Evaluate on test set
print("Test Results:")
y_test_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Summary
print("\n")
print(f"Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")