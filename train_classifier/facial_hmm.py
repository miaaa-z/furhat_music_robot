import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hmmlearn import hmm

print("Training Facial Expression Classifier (HMM)\n")


FEATURES_PATH = '/Users/miaaa/Desktop/music robot/furhat_music_robot/features_v3.csv'
METADATA_PATH = '/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv'


f2 = pd.read_csv(FEATURES_PATH)
m2 = pd.read_csv(METADATA_PATH)
f2['facial_expression'] = f2['facial_expression'].str.strip()
f2['split'] = f2['song_name'].map(dict(zip(m2['title'], m2['split'])))


def merge_facial(expr):
    e = str(expr).strip().lower()
    if 'angry' in e or 'disgust' in e:
        return 'negative'
    if 'big smile' in e or ('browraise' in e and 'smile' in e):
        return 'big_smile'
    if 'browraise' in e or 'surprise' in e or 'oh' in e:
        return 'surprise'
    if 'frown' in e or 'sad' in e:
        return 'frown'
    if 'thoughtful' in e or 'confused' in e:
        return 'thoughtful'
    if 'smile' in e:
        return 'smile'
    return 'neutral'


f2['facial_merged'] = f2['facial_expression'].apply(merge_facial)

# Merge
facial_merge = {
    'negative':   'frown',
    'thoughtful': 'neutral',
    'surprise':   'big_smile',
}
f2['facial_label'] = f2['facial_merged'].replace(facial_merge)
FACIAL_CLASSES = sorted(f2['facial_label'].unique())
print('Facial classes:', FACIAL_CLASSES)


ftr = f2[f2['split'] == 'Train']
fva = f2[f2['split'] == 'Validation']
fte = f2[f2['split'] == 'Test']


LABEL_COLS = ['song_name', 'start_time', 'end_time', 'duration',
              'head_movement', 'facial_expression', 'intensity',
              'facial_merged', 'facial_label', 'split']
ALL_FEATS = [c for c in f2.columns if c not in LABEL_COLS]

mfcc_feats = [c for c in ALL_FEATS if c.startswith('mfcc') and 'delta' not in c]
chroma_feats = [c for c in ALL_FEATS if c.startswith('chroma')]
BEST_FEATS = mfcc_feats + chroma_feats   # mfcc+chroma，Val=0.3906 best
BEST_N_COMP = 2                            # n_comp=2

print(f"Feature set: mfcc+chroma ({len(BEST_FEATS)} features), n_comp={BEST_N_COMP}\n")


scaler = StandardScaler()
X_train = scaler.fit_transform(ftr[BEST_FEATS].values)
X_val = scaler.transform(fva[BEST_FEATS].values)
X_test = scaler.transform(fte[BEST_FEATS].values)

y_train = ftr['facial_label'].values
y_val = fva['facial_label'].values
y_test = fte['facial_label'].values

models = {}
for cls in FACIAL_CLASSES:
    segs = X_train[y_train == cls]
    if len(segs) < BEST_N_COMP:
        continue
    k = min(BEST_N_COMP, len(segs))
    m = hmm.GaussianHMM(n_components=k, covariance_type='diag',
                        n_iter=200, random_state=42, verbose=False)
    m.fit(segs, [len(segs)])
    models[cls] = m
    print(f"  Trained HMM for class: {cls} ({len(segs)} samples)")


def predict(X):
    preds = []
    for x in X:
        obs = x.reshape(1, -1)
        best_cls, best_score = None, -np.inf
        for cls, model in models.items():
            try:
                s = model.score(obs)
                if s > best_score:
                    best_score, best_cls = s, cls
            except:
                pass
        preds.append(best_cls)
    return np.array(preds)


val_acc = accuracy_score(y_val,  predict(X_val))
test_acc = accuracy_score(y_test, predict(X_test))
print(f"\nVal Accuracy:  {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

SAVE_MODEL = False

if SAVE_MODEL:
    save_dir = '/Users/miaaa/Desktop/music robot/furhat_music_robot/models'
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(models,     f'{save_dir}/facial_hmm_models.pkl')
    joblib.dump(scaler,     f'{save_dir}/facial_hmm_scaler.pkl')
    joblib.dump(BEST_FEATS, f'{save_dir}/facial_hmm_feature_cols.pkl')

    print(f"\nModel saved to {save_dir}")
    print("  facial_hmm_models.pkl")
    print("  facial_hmm_scaler.pkl")
    print("  facial_hmm_feature_cols.pkl")