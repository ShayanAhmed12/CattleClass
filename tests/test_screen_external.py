import numpy as np

from screen_external import decision


def test_screen_requires_two_views_with_same_confident_reason():
    labels = ("valid", "drawing", "text")
    confident = [0.20, 0.28, 0.22]
    assert decision(np.array([confident, confident]), labels) == "drawing"
    assert not decision(np.array([confident, [0.20, 0.22, 0.28]]), labels)
    assert not decision(np.array([confident, [0.24, 0.28, 0.20]]), labels)
    assert not decision(np.array([confident, [0.20, 0.28, 0.275]]), labels)
    assert not decision(np.array([[0.28, 0.21, 0.20]] * 2), labels)
