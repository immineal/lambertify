"""Audio processing utilities: spectrogram conversion and Griffin-Lim vocoding."""
import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
DB_MIN = -80.0
DB_MAX = 0.0

CHUNK_DURATION = 4.0
FRAMES_PER_CHUNK = int(CHUNK_DURATION * SAMPLE_RATE / HOP_LENGTH)  # 172


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def audio_to_melspec(audio: np.ndarray) -> np.ndarray:
    """Waveform → log-mel spectrogram in dB, clipped to [DB_MIN, DB_MAX]."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.clip(mel_db, DB_MIN, DB_MAX)


def normalize(mel_db: np.ndarray) -> np.ndarray:
    """Map [DB_MIN, DB_MAX] → [-1, 1]."""
    return 2.0 * (mel_db - DB_MIN) / (DB_MAX - DB_MIN) - 1.0


def denormalize(mel_norm: np.ndarray) -> np.ndarray:
    """Map [-1, 1] → [DB_MIN, DB_MAX]."""
    return (mel_norm + 1.0) / 2.0 * (DB_MAX - DB_MIN) + DB_MIN


def melspec_to_audio(mel_db: np.ndarray, n_iter: int = 64) -> np.ndarray:
    """Log-mel spectrogram → waveform via Griffin-Lim."""
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
        n_iter=n_iter,
    )
    return audio