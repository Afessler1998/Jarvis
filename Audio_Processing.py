import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

def process_audio(file_path):
    # Load the .wav file with its native sample rate
    y, sr = librosa.load(file_path, sr=None)

    # Trim silence and normalize audio
    trimmed_audio, _ = librosa.effects.trim(y, top_db=20)
    normalized_audio = librosa.util.normalize(trimmed_audio)

    # Convert to NumPy array for noise reduction
    samples = np.array(normalized_audio)

    # Reduce noise
    reduced_noise = nr.reduce_noise(samples, sr=sr)

    # Save the processed audio back to the same file
    sf.write(file_path, reduced_noise, sr, subtype='PCM_16')