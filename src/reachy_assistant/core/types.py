from dataclasses import dataclass

@dataclass(frozen=True)
class Transcript:
    text: str

@dataclass(frozen=True)
class Answer:
    text: str