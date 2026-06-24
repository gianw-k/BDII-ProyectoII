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
vector de longitud fija (2*n_mfcc = 40 por defecto con n_mfcc=20),
exactamente igual que las columnas mfcc1_mean..mfcc20_var del CSV GTZAN.
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

    def _from_audio(self, source: Union[str, Path, bytes]) -> np.ndarray:
        """Extrae features MFCC de un archivo de audio (ruta o bytes)."""
        try:
            import librosa  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "librosa no disponible. Instala 'librosa' para extraer MFCC desde audio."
            ) from e

        if isinstance(source, (bytes, bytearray)):
            import io
            y, _ = librosa.load(io.BytesIO(bytes(source)), sr=self.sr, mono=True)
        else:
            y, _ = librosa.load(str(source), sr=self.sr, mono=True)

        # Extraer n_mfcc coeficientes a lo largo del tiempo
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)  # (n_mfcc, T)

        # Reducir a un solo vector de longitud fija: [mean_1..mean_n, var_1..var_n]
        mean = mfcc.mean(axis=1)   # (n_mfcc,)
        var = mfcc.var(axis=1)     # (n_mfcc,)
        feat = np.concatenate([mean, var]).astype(np.float32)  # (2*n_mfcc,)

        return feat.reshape(1, -1)  # (1, 2*n_mfcc)
