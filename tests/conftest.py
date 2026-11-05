"""Lets the tests import fccl.py, cdm.py and live_cascade_test.py from the repo root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))