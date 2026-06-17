#!/usr/bin/env python
"""Deprecated single-file dashboard.

This file has been replaced by the modular app/ package.
Run the new entry point instead::

    streamlit run app/main.py

The old code is preserved in git history. See commit d5d2e80 and earlier.
"""
import sys
import os

if __name__ == "__main__":
    print(__doc__)
    # Point the user at the new entry point
    entry = os.path.join(os.path.dirname(__file__), "app", "main.py")
    if os.path.exists(entry):
        print(f"\nLaunch with: streamlit run {entry}")
    sys.exit(0)