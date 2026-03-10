from reachy_mini import ReachyMini
from reachy_assistant.tts.piper_tts import PiperTTSEngine
from reachy_assistant.robot.audio.play_wav import play_on_reachy

tts = PiperTTSEngine(model_path="models/no_NO-talesyntese-medium.onnx")

wav = tts.synthesize_wav_bytes("Hei! Dette er en test, hvordan høres det ut?.")

with ReachyMini() as mini:
    play_on_reachy(mini, wav)