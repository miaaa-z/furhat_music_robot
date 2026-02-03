import pickle
import pandas as pd

# Load dataset
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Convert to DataFrame
data = {
    'Tempo': [features[0] for features, _ in dataset],
    'Brightness': [features[1] for features, _ in dataset],
    'Energy': [features[2] for features, _ in dataset],
    'Label': [label for _, label in dataset]
}

df = pd.DataFrame(data)

# Save to Excel
df.to_excel('dataset.xlsx', index=False)
print("Saved to dataset.xlsx")

# Show preview
print("\nFirst 10 rows:")
print(df.head(10))