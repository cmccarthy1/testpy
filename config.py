"""Configuration values for building/packaging."""

from pathlib import Path
import platform

script_dir = Path(__file__).parent.absolute()

pypy = platform.python_implementation() == 'PyPy'

debug = True

with open(script_dir/'src'/'qtestpy'/'VERSION') as version_file:
    version = version_file.read().strip()
