# -*- coding: utf-8 -*-
"""Ensure CLI scripts load opencood from the repository that contains them."""
import os
import sys


def prepend_repo_root():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
