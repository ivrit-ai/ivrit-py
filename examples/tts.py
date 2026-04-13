import soundfile as sf
from ivrit.tts import TTS

tts = TTS()  # models auto-download on first run

audio, sr = tts.synthesize("שימו לב נוסעים יקרים, הרכבת תכנס לתחנת תל אביב מרכז בעוד מספר דקות.")
sf.write("output.wav", audio, sr)
print("Saved output.wav")
