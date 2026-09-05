"""Put src/ on the import path so the tests run from a clean checkout.

The alternative is `pip install -e .`, which works too, but a reader who has
just cloned the repository should be able to type `python3 -m pytest -q` and
have it pass with nothing else installed but the requirements.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
