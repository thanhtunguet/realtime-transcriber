"""Lightweight online speaker diarization.

Each audio chunk is reduced to a small voice fingerprint (average MFCC envelope
plus median pitch) and assigned to the nearest known speaker. A new speaker is
only created when the chunk is clearly unlike every known speaker and the next
chunk agrees, so the number of speakers can grow freely while false splits stay
rare. Similar-sounding voices are merged rather than split (conservative).
"""
import numpy as np
import librosa

# Rough per-coefficient spread of MFCC 1..19 across speakers, used to put all
# coefficients on a comparable scale before measuring distance.
_MFCC_SCALE = np.array(
    [44.5, 35.7, 32.5, 25.5, 23.8, 21.1, 16.6, 17.0, 13.1, 14.0,
     13.9, 16.7, 11.1, 11.5, 11.3, 10.1, 10.3, 9.2, 9.2],
    dtype=np.float32,
)
_PITCH_WEIGHT = 4.0  # weight of log-pitch relative to one scaled MFCC unit


class OnlineDiarizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        new_speaker_distance: float = 1.5,
        confirm_chunks: int = 2,
        max_speakers: int = 8,
        min_voiced_seconds: float = 0.6,
    ):
        """
        new_speaker_distance: a chunk must be at least this far from every
            known speaker to count as someone new. Higher is more conservative
            (fewer speakers); same-speaker chunks typically land around 0.4-1.1.
        confirm_chunks: consecutive "unknown" chunks that must resemble each
            other before a new speaker is created, so a cough or a noisy chunk
            doesn't spawn a phantom speaker.
        """
        self.sample_rate = sample_rate
        self.new_speaker_distance = new_speaker_distance
        self.confirm_chunks = max(1, confirm_chunks)
        self.max_speakers = max_speakers
        self.min_voiced_seconds = min_voiced_seconds

        self._sums: list[np.ndarray] = []  # running sum of fingerprints per speaker
        self._counts: list[int] = []
        self._pending: list[np.ndarray] = []  # recent unknown-looking chunks
        self._last_speaker: int | None = None

    @property
    def num_speakers(self) -> int:
        return len(self._counts)

    def _fingerprint(self, audio: np.ndarray) -> np.ndarray | None:
        """Voice fingerprint of the voiced part of a chunk, or None if too little speech."""
        hop = 160
        rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop)[0]
        if rms.max() <= 0:
            return None
        voiced = rms > max(rms.max() * 0.2, 1e-4)

        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=400, hop_length=hop
        )
        f0 = librosa.yin(
            audio, fmin=70, fmax=350, sr=self.sample_rate, frame_length=1024, hop_length=hop
        )
        n = min(mfcc.shape[1], len(f0), len(voiced))
        voiced = voiced[:n]
        if voiced.sum() * hop / self.sample_rate < self.min_voiced_seconds:
            return None

        # Skip c0 (loudness), so volume and mic distance don't matter.
        envelope = mfcc[1:, :n][:, voiced].mean(axis=1) / _MFCC_SCALE
        pitch = np.log(np.median(f0[:n][voiced])) * _PITCH_WEIGHT
        return np.append(envelope, pitch)

    def _centroid(self, i: int) -> np.ndarray:
        return self._sums[i] / self._counts[i]

    def identify(self, audio: np.ndarray) -> int | None:
        """Return the 0-based speaker index for this chunk (None if undecidable)."""
        feat = self._fingerprint(audio)
        if feat is None:
            return self._last_speaker

        if not self._counts:
            self._sums.append(feat.copy())
            self._counts.append(1)
            self._last_speaker = 0
            return 0

        dists = [float(np.linalg.norm(feat - self._centroid(i))) for i in range(self.num_speakers)]
        best = int(np.argmin(dists))

        if dists[best] < self.new_speaker_distance or self.num_speakers >= self.max_speakers:
            self._pending.clear()
            self._sums[best] = self._sums[best] + feat
            self._counts[best] += 1
            self._last_speaker = best
            return best

        # Looks like someone new: wait until consecutive chunks agree. Until
        # then the chunk is left unlabeled rather than blamed on someone else.
        self._pending.append(feat)
        if len(self._pending) >= self.confirm_chunks:
            mean = np.mean(self._pending, axis=0)
            if all(np.linalg.norm(p - mean) < self.new_speaker_distance for p in self._pending):
                self._sums.append(np.sum(self._pending, axis=0))
                self._counts.append(len(self._pending))
                self._pending.clear()
                self._last_speaker = self.num_speakers - 1
                return self._last_speaker
            self._pending = self._pending[-1:]  # disagreed: keep only the latest
        return None
