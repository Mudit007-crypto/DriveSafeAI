import sys
import numpy as np
import librosa
import requests
import json

def wav_to_mel_spectrogram_uint8(file_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512, fixed_size=(128,128)):
    y, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    mel_255 = (mel_norm * 255).astype(np.uint8)

    mel_resized_time = librosa.util.fix_length(mel_255, size=fixed_size[1], axis=1, mode='constant')
    current_height = mel_resized_time.shape[0]
    if current_height < fixed_size[0]:
        pad_amount = fixed_size[0] - current_height
        mel_resized = np.pad(mel_resized_time, ((0, pad_amount), (0,0)), mode='constant')
    else:
        mel_resized = mel_resized_time[:fixed_size[0], :]

    return mel_resized

def classify_audio(filepath):
    mel = wav_to_mel_spectrogram_uint8(filepath)
    mel = np.stack([mel]*3, axis=-1)  # Make it (128,128,3)
    mel = mel.astype(np.float32) / 255.0  # Normalize to [0, 1]
    mel = np.expand_dims(mel, axis=0)  # Add batch dim => (1, 128, 128, 3)

    payload = {
        "inputs": [{
            "name": "input",
            "shape": mel.shape,
            "datatype": "FP32",
            "data": mel.flatten().tolist()
        }]
    }

    try:
        response = requests.post("http://localhost:8000/v2/models/studio_classifier/infer", json=payload)
        response.raise_for_status()
        result = response.json()
        probs = np.array(result['outputs'][0]['data'])
        predicted_class = np.argmax(probs)
        print(f"✅ Predicted class: {predicted_class} (raw: {probs})")
    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_cl.py <path_to_wav_file>")
        sys.exit(1)
    classify_audio(sys.argv[1])