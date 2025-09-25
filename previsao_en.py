import os
import sys
import joblib
import librosa
import numpy as np
from pydub import AudioSegment

def convert_to_wav(audio_path, output_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Erro ao converter o arquivo de áudio {audio_path} para .wav: {e}")
        return None

def extract_features(audio_path, n_mfcc=20, max_n_fft=2048, hop_length=512):
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

def analyze_audio(input_audio_path):
    if not os.path.exists(input_audio_path):
        print(f"Erro: O arquivo '{input_audio_path}' não existe.")
        return
    
    if not input_audio_path.lower().endswith(".wav"):
        print(f"Convertendo o arquivo '{input_audio_path}' para .wav...")
        wav_path = os.path.splitext(input_audio_path)[0] + ".wav"
        input_audio_path = convert_to_wav(input_audio_path, wav_path)
        if input_audio_path is None:
            return
    
    model_filename = "xgb_model.pkl"  
    scaler_filename = "scaler.pkl"
    pca_filename = "pca.pkl"
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename) or not os.path.exists(pca_filename):
        print("Erro: Modelo, scaler ou PCA não encontrados. Treine o modelo primeiro.")
        return
    
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    pca = joblib.load(pca_filename)

    features = extract_features(input_audio_path)

    if features is not None:
        features_pca = pca.transform(features.reshape(1, -1))
        features_scaled = scaler.transform(features_pca)
        prediction = model.predict(features_scaled)
        if prediction[0] == 0:
            print("O áudio de entrada foi classificado como genuíno.")
        else:
            print("O áudio de entrada foi classificado como deepfake.")
    else:
        print("Erro: Não foi possível processar o áudio de entrada.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python analyze.py <caminho_do_arquivo>")
    else:
        audio_path = sys.argv[1]
        analyze_audio(audio_path)