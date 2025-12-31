import pytest
from metrics import precision_at_k, recall_at_k

def test_precision_and_recall():
    true = [1,2,3]
    pred = [3,4,1,5]
    p = precision_at_k(true, pred, k=3)
    r = recall_at_k(true, pred, k=3)
    assert pytest.approx(p, rel=1e-3) == 2/3
    assert pytest.approx(r, rel=1e-3) == 2/3
