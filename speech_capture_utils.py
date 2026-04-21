import io
import wave
import numpy as np

def pcm_to_wav_bytes(pcm_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data.tobytes())
    return buffer.getvalue()

class InMemoryAudioFile:
    def __init__(self, wav_bytes):
        self.buffer = io.BytesIO(wav_bytes)
    
    def read(self):
        return self.buffer.read()
