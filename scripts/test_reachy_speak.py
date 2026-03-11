from reachy_mini import ReachyMini
from reachy_assistant.tts.piper_tts import PiperTTSEngine
from reachy_assistant.robot.audio.reachy_player import ReachyPlayer
tts = PiperTTSEngine(model_path="models/no_NO-talesyntese-medium.onnx")

wav = tts.synthesize_wav_bytes("Hei! Dette er en test, hvordan høres det ut?.")

with ReachyMini() as mini:
    player = ReachyPlayer(mini)
    player.play_wav(wav)