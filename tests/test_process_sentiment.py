import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from process_sentiment import calculate_unified_score

def test_calculate_unified_score():
    pos = 0.8
    neg = 0.1
    neu = 0.1
    unified = calculate_unified_score(pos, neg, neu)
    assert round(unified, 2) == 0.7  # 0.8 - 0.1

    pos2 = 0.2
    neg2 = 0.7
    neu2 = 0.1
    unified2 = calculate_unified_score(pos2, neg2, neu2)
    assert round(unified2, 2) == -0.5
