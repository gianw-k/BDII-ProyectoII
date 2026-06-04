"""Hace importable el paquete `app` al correr pytest desde la raiz del repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
