import pandas as pd
from collections import Counter

# Read behaviors from each sheet separately
sheet_names = {
    'Sheet1': 'Someone Like You',
    'Sheet2': 'Uptown Funk',
    'Sheet3': 'Back to Black'
}

all_behaviors = []
song_behaviors = {}  # Store behaviors for each song

for sheet_id, song_name in sheet_names.items():
    df = pd.read_excel('3_songs.xlsx', sheet_name=sheet_id)
    e_column = df.iloc[:, 4].dropna()  # Column E - head movement
    f_column = df.iloc[:, 5].dropna()  # Column F - facial expression

    song_behaviors[song_name] = []
    song_behaviors[song_name].extend(e_column.tolist())
    song_behaviors[song_name].extend(f_column.tolist())

    all_behaviors.extend(e_column.tolist())
    all_behaviors.extend(f_column.tolist())


def normalize_behavior(behavior_str):
    # Basic cleanup
    behavior_str = str(behavior_str).lower().strip()
    return behavior_str


def split_and_count(behaviors_list):
    # Simply split by comma and count
    split_behaviors = []

    for behavior in behaviors_list:
        behavior_str = str(behavior)

        # If contains comma, split it
        if ',' in behavior_str:
            parts = behavior_str.split(',')
            for part in parts:
                part = normalize_behavior(part)
                if part and part != 'nan':
                    split_behaviors.append(part)
        else:
            normalized = normalize_behavior(behavior_str)
            if normalized and normalized != 'nan':
                split_behaviors.append(normalized)

    return Counter(split_behaviors)


# Print behaviors for each song
song_counters = {}
for song_name, behaviors in song_behaviors.items():
    print(f"\n{song_name}:")
    print("-" * 70)

    counter = split_and_count(behaviors)
    song_counters[song_name] = counter

    sorted_behaviors = sorted(counter.items(), key=lambda x: -x[1])

    for behavior, count in sorted_behaviors:
        print(f"  {behavior:35} {count:3}x")

    print(f"\n  Total unique behaviors: {len(counter)}")
    print(f"  Total instances: {sum(counter.values())}")

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)

behavior_counts = split_and_count(all_behaviors)
sorted_counts = sorted(behavior_counts.items(), key=lambda x: -x[1])

print(f"\nTotal Unique Behaviors: {len(behavior_counts)}")
print(f"Total Instances: {sum(behavior_counts.values())}")

print("\nBehavior Frequencies (All Songs Combined):")
print("-" * 70)

for behavior, count in sorted_counts:
    print(f"{behavior:35} {count:3}x")

# Low frequency behaviors
print("\nLow-Frequency Behaviors (< 3 times):")
print("-" * 70)
low_freq = [(b, c) for b, c in sorted_counts if c < 3]
if low_freq:
    for behavior, count in low_freq:
        print(f"  {behavior}: {count} time(s)")
else:
    print("  None")

