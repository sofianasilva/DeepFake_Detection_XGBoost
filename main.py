import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

def extract_features(audio_path, n_mfcc=20, max_n_fft=512, hop_length=256):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Erro ao carregar o arquivo de áudio {audio_path}: {e}")
        return None

    n_fft = min(max_n_fft, len(audio_data))
    if n_fft < max_n_fft:
        print(f"Aviso: n_fft ajustado para {n_fft} no arquivo {audio_path} (comprimento do áudio: {len(audio_data)})")

    if len(audio_data) < n_fft:
        audio_data = librosa.util.fix_length(audio_data, size=n_fft)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_mean = np.mean(chroma.T, axis=0)

    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate.T, axis=0)

    features = np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean, zero_crossing_rate_mean))
    return features

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))  
    for audio_path in audio_files:
        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            print(f"Ignorando arquivo de áudio {audio_path}")

    print("Número de amostras em", directory, ":", len(X))
    return X, y

def train_model(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    pca = PCA(n_components=30, random_state=42)
    X_pca = pca.fit_transform(X_balanced)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print("Acurácia:", accuracy)
    print("Matriz de Confusão:")
    print(confusion_mtx)
    print("Relatório de Classificação:")
    print(classification_rep)

    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")

def main():
    genuine_dir = "C:/Users/sofia/OneDrive/Documentos/Juncao_modelos/DATASET_INGLES/REAL"
    deepfake_dir = "C:/Users/sofia/OneDrive/Documentos/Juncao_modelos/DATASET_INGLES/FAKE"

    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

    X = np.vstack((X_genuine, X_deepfake))
    y = np.hstack((y_genuine, y_deepfake))

    train_model(X, y)

if __name__ == "__main__":
    main()