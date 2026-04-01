from reachy_mini import ReachyMini

with ReachyMini(media_backend="no_media") as mini:
    print("connected")
    input("press enter to exit")