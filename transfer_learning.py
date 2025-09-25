from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import librosa
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

REAL_DIR = "C:/Users/Usuario/Documents/xgboost/dataset/Sulista/REAL"
FAKE_DIR = "C:/Users/Usuario/Documents/xgboost/dataset/Sulista/FAKE"
OUTPUT_DIR = "C:/Users/Usuario/Documents/xgboost/dataset/Sulista/models"
RANDOM_STATE = 42
TEST_SIZE = 0.2  


def load_and_extract_features(file_list, feature_extractor, model, label):
    features = []
    for file in tqdm(file_list, desc=f"Extraindo {label}"):
        audio, sr = librosa.load(file, sr=16000)
        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        feat = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        features.append(feat)
    return features


def main():
    real_files = glob.glob(os.path.join(REAL_DIR, "*.wav"))
    fake_files = glob.glob(os.path.join(FAKE_DIR, "*.wav"))
    print(f"Encontrados {len(real_files)} reais e {len(fake_files)} fakes")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    real_features = load_and_extract_features(real_files, feature_extractor, wav2vec_model, "REAL")
    fake_features = load_and_extract_features(fake_files, feature_extractor, wav2vec_model, "FAKE")

    X = np.vstack([real_features, fake_features])
    y = np.array([0]*len(real_features) + [1]*len(fake_features))
    print(f"Formato X: {X.shape} y: {y.shape}")
    print("Distribuição original:", Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Tamanhos -> treino: {len(y_train)}  teste: {len(y_test)}")
    print("Distribuição treino:", Counter(y_train), " teste:", Counter(y_test))

    pca = PCA(n_components=30)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print("Train pos/neg:", Counter(y_train), " scale_pos_weight:", scale_pos_weight)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
    }

    model = xgb.train(params, dtrain, num_boost_round=200)

    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_model(os.path.join(OUTPUT_DIR, "xgb_model.json"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca.pkl"))
    print(f"Modelos salvos em {OUTPUT_DIR}")

    cm = confusion_matrix(y_test, y_pred)
    labels = ["REAL", "FAKE"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - XGBoost Southern")
    plt.show()

if __name__ == "__main__":
    main()
