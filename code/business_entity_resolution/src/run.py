#!/usr/bin/env python3
"""
Entry point for Business Entity Resolution pipeline.
Run: python -m src.run
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import main

if __name__ == '__main__':
    main()