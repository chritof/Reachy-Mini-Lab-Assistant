from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline


class DummySTT:
    def transcribe(self):
        return input("You: ")


class DummyRAG:
    def ask(self, question):
        return f"I heard: {question}"


class DummyTTS:
    def speak(self, text):
        print("Robot:", text)


class DummyMotion:
    def idle(self):
        print("Motion idle")

    def listening(self):
        print("Motion listening")

    def thinking(self):
        print("Motion thinking")

    def speaking(self):
        print("Motion speaking")


def main():
    pipeline = ConversationPipeline(
        stt=DummySTT(),
        rag=DummyRAG(),
        tts=DummyTTS(),
        motion=DummyMotion(),
    )

    pipeline.run_forever()


if __name__ == "__main__":
    main()