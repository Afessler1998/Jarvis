import os
import pyttsx3
import soundfile
from speechbrain.pretrained import SpeakerRecognition
import speech_recognition
import torch
import whisper


# Text to speech voice
_voice = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\CereVoice William 6.1.0'

# Speech to text model
whisper.load_model('base')

# Speaker recognition model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models")


def record_audio():
    try:
        recognizer = speech_recognition.Recognizer()
        #recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 2500
        recognizer.pause_threshold = 2
        with speech_recognition.Microphone() as source:
            audio = recognizer.listen(source)
            return audio
    except Exception as e:
        print(f"Audio recording error: {e}")
        return None
    

def wav_to_audio(file_path):
    recognizer = speech_recognition.Recognizer()
    with speech_recognition.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        return audio


def speech_to_text(audio):
    try:
        recognizer = speech_recognition.Recognizer()
        return recognizer.recognize_whisper(audio)
    except speech_recognition.WaitTimeoutError:
        return None
    except speech_recognition.UnknownValueError:
        print("Audio could not be transcribed")
        return None

    
def text_to_speech(text):
    try:
        audio_engine = pyttsx3.init()
        audio_engine.setProperty('voice', _voice)
        audio_engine.say(text)
        audio_engine.runAndWait()
        audio_engine.stop()
    except Exception as e:
        print(f"Text to speech error: {e}")


def recognize_speaker(input_audio_path):
    # Thresholds
    high_similarity_threshold = 0.9
    lower_similarity_threshold = 0.75

    # Directory containing user voice samples
    user_voices_dir = 'user_voices'

    # Process the input audio to get its embedding
    input_signal, _ = soundfile.read(input_audio_path)
    input_embedding = model.encode_batch(input_signal).squeeze(0)

    # Initialize variables to track the most likely speaker
    highest_similarity = 0
    most_likely_speaker = None

    # Process and compare each user's voice sample immediately
    for filename in os.listdir(user_voices_dir):
        if filename.endswith('.wav'):
            user_name = filename.split('.')[0]
            signal, _ = soundfile.read(os.path.join(user_voices_dir, filename))
            user_embedding = model.encode_batch(signal).squeeze(0)

            similarity = torch.nn.functional.cosine_similarity(input_embedding, user_embedding, dim=0)

            # Early return if a user exceeds the high threshold
            if similarity > high_similarity_threshold:
                return user_name

            # Keep track of the most likely speaker above the lower threshold
            if similarity > highest_similarity and similarity > lower_similarity_threshold:
                highest_similarity = similarity
                most_likely_speaker = user_name

    return most_likely_speaker
