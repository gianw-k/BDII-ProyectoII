"""Extractor MFCC para audio (Mel-Frequency Cepstral Coefficients).

Acepta dos formas de entrada:
  1. Array numpy (1-D o 2-D): vector de features ya pre-calculadas (viene del CSV).
     Se devuelve directamente como (1, n_features) sin tocar librosa.
  2. str / Path / bytes: ruta o bytes de un archivo .wav/.mp3.
     Extrae n_mfcc coeficientes con librosa, calcula mean y var por coeficiente
     y devuelve un unico vector (1, 2*n_mfcc) con la misma forma que el CSV.

Esto permite que el pipeline de ingest use el CSV de features ya calculadas
(rapido, sin leer los .wav) y que el endpoint online procese el audio que
sube el usuario en tiempo real (mas lento pero correcto academicamente).

El extractor es invariante a la duracion del clip: siempre produce un
vector de longitud fija (57 dimensiones), exactamente igual
que las columnas extraidas del CSV GTZAN de 3 segundos.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np

from app.engine.base import Extractor

# Dimensionalidad por defecto = 20 MFCCs x 2 (mean+var) = 40, igual que el CSV GTZAN
_N_MFCC = 20
_SR = 22050          # sample rate estandar de librosa


class MFCCExtractor(Extractor):
    """Extractor de palabras acusticas MFCC.

    Parameters
    ----------
    n_mfcc: int
        Numero de coeficientes MFCC a calcular. Por defecto 20 (igual que GTZAN).
    sr: int
        Sample rate al que se resamplea el audio. Ignorado si la entrada ya
        es un array numpy de features pre-calculadas.
    """

    def __init__(self, n_mfcc: int = _N_MFCC, sr: int = _SR) -> None:
        self.n_mfcc = n_mfcc
        self.sr = sr

    def extract(self, chunk: Union[np.ndarray, str, Path, bytes]) -> np.ndarray:
        """chunk -> array (1, 2*n_mfcc) de features MFCC (mean + var).

        Si chunk ya es un numpy array se devuelve reshaped a (1, n).
        Si es ruta/bytes se procesa con librosa.
        """
        if isinstance(chunk, np.ndarray):
            return np.atleast_2d(chunk.astype(np.float32))

        return self._from_audio(chunk)

    def _extract_57_features(self, y_seg: np.ndarray, librosa) -> np.ndarray:
        # Extraer caracteristicas adicionales de librosa
        chroma = librosa.feature.chroma_stft(y=y_seg, sr=self.sr)
        rms = librosa.feature.rms(y=y_seg)
        spec_cent = librosa.feature.spectral_centroid(y=y_seg, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y_seg, sr=self.sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=self.sr)
        zcr = librosa.feature.zero_crossing_rate(y=y_seg)
        y_harm, y_perc = librosa.effects.hpss(y_seg)
        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=self.sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

        # Extraer MFCCs
        mfcc = librosa.feature.mfcc(y=y_seg, sr=self.sr, n_mfcc=self.n_mfcc)

        features = [
            chroma.mean(), chroma.var(),
            rms.mean(), rms.var(),
            spec_cent.mean(), spec_cent.var(),
            spec_bw.mean(), spec_bw.var(),
            rolloff.mean(), rolloff.var(),
            zcr.mean(), zcr.var(),
            y_harm.mean(), y_harm.var(),
            y_perc.mean(), y_perc.var(),
            tempo_val
        ]

        mfcc_mean = mfcc.mean(axis=1)
        mfcc_var = mfcc.var(axis=1)

        feat = np.concatenate([
            np.array(features, dtype=np.float32),
            mfcc_mean.astype(np.float32),
            mfcc_var.astype(np.float32)
        ])
        return feat

    def _from_audio(self, source: Union[str, Path, bytes]) -> np.ndarray:
        """Extrae features MFCC y demas espectrales de un archivo de audio (ruta o bytes)."""
        try:
            import librosa  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "librosa no disponible. Instala 'librosa' para extraer caracteristicas de audio."
            ) from e

        if isinstance(source, (bytes, bytearray)):
            import io
            y, _ = librosa.load(io.BytesIO(bytes(source)), sr=self.sr, mono=True)
        else:
            y, _ = librosa.load(str(source), sr=self.sr, mono=True)

        window_size = int(3 * self.sr)
        if len(y) <= window_size:
            # Clip corto: un solo segmento
            feat = self._extract_57_features(y, librosa)
            return feat.reshape(1, -1)

        # Clip largo: dividir en ventanas de 3 segundos sin solapamiento
        feats = []
        for start in range(0, len(y), window_size):
            end = start + window_size
            # Si el ultimo fragmento es menor a 0.5s, lo ignoramos para evitar ruido
            if end > len(y) and (len(y) - start) < int(0.5 * self.sr):
                continue
            y_seg = y[start:min(end, len(y))]
            feat = self._extract_57_features(y_seg, librosa)
            feats.append(feat)

        return np.stack(feats)

