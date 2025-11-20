import pandas as pd

class GTZANAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.key_features = [
            'tempo',
            'rms_mean',
            'spectral_centroid_mean',
            'spectral_bandwidth_mean',
            'zero_crossing_rate_mean',
            'rolloff_mean'
        ]

    def analyze_by_genre(self):
        genres = self.df['label'].unique()
        stats_list = []
        for genre in sorted(genres):
            g = self.df[self.df['label'] == genre]
            stats = {'Genre': genre.capitalize()}
            for feature in self.key_features:
                stats[f'{feature}_min'] = g[feature].min()
                stats[f'{feature}_max'] = g[feature].max()
                stats[f'{feature}_mean'] = g[feature].mean()
                stats[f'{feature}_std'] = g[feature].std()
            stats_list.append(stats)
        return pd.DataFrame(stats_list)

    def print_feature_ranges(self, stats_df):
        for _, row in stats_df.iterrows():
            print(f"\n{row['Genre']}")
            print("-" * len(row['Genre']))
            print(f"Tempo: {row['tempo_min']:.1f} - {row['tempo_max']:.1f} (mean {row['tempo_mean']:.1f})")
            print(f"Energy (rms): {row['rms_mean_min']:.3f} - {row['rms_mean_max']:.3f} (mean {row['rms_mean_mean']:.3f})")
            print(f"Brightness (centroid): {row['spectral_centroid_mean_min']:.0f} - {row['spectral_centroid_mean_max']:.0f} (mean {row['spectral_centroid_mean_mean']:.0f})")
            print(f"Bandwidth: {row['spectral_bandwidth_mean_min']:.0f} - {row['spectral_bandwidth_mean_max']:.0f} (mean {row['spectral_bandwidth_mean_mean']:.0f})")
            print(f"Noisiness (ZCR): {row['zero_crossing_rate_mean_min']:.3f} - {row['zero_crossing_rate_mean_max']:.3f} (mean {row['zero_crossing_rate_mean_mean']:.3f})")
            print(f"Rolloff: {row['rolloff_mean_min']:.0f} - {row['rolloff_mean_max']:.0f} (mean {row['rolloff_mean_mean']:.0f})")

def main():
    csv_path = "/Users/miaaa/Desktop/music robot/furhat_music_robot/data/raw/Data/features_30_sec.csv"
    analyzer = GTZANAnalyzer(csv_path)
    stats_df = analyzer.analyze_by_genre()
    analyzer.print_feature_ranges(stats_df)

if __name__ == "__main__":
    main()
