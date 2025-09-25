import os
import sys
import joblib
import librosa
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def convert_to_wav(audio_path, output_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Erro ao converter o arquivo de áudio {audio_path} para .wav: {e}")
        return None

def extract_wav2vec_features(audio_path):
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Erro ao carregar o arquivo de áudio {audio_path}: {e}")
        return None

    inputs = feature_extractor(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  
    return embeddings.flatten()

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
    
    model_filename = "xgb_model_transfer_learning_portuguese.pkl"
    
    if not os.path.exists(model_filename):
        print("Erro: Modelo não encontrado. Treine o modelo primeiro.")
        return
    
    model = joblib.load(model_filename)

    features = extract_wav2vec_features(input_audio_path)

    if features is not None:
        prediction = model.predict(features.reshape(1, -1))
        if prediction[0] == 0:
            print("O áudio de entrada foi classificado como genuíno (REAL).")
        else:
            print("O áudio de entrada foi classificado como deepfake (FAKE).")
    else:
        print("Erro: Não foi possível processar o áudio de entrada.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python previsao_pt.py <caminho_do_arquivo>")
    else:
        audio_path = sys.argv[1]
        analyze_audio(audio_path)