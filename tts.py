"""Background-thread TTS wrapper around pyttsx3 (Windows SAPI5)."""
from __future__ import annotations
import queue
import threading
import time
import pyttsx3


class TTSWorker:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # Engine MUST be constructed in the worker thread on Windows.
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        while True:
            msg = self._q.get()
            if msg is None:
                break
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception as e:
                print(f"[tts] error: {e}")

    def speak_async(self, text: str):
        # Drop if backlog is large — don't queue up stale corrections.
        if self._q.qsize() > 1:
            return
        self._q.put(text)

    def stop(self):
        self._q.put(None)


if __name__ == "__main__":
    tts = TTSWorker()
    tts.speak_async("Yoga pose correction system online.")
    time.sleep(3)
    tts.stop()
