import pytest
import microsim


def test_random():
    """
    Checks that random classes are created properly
    :return:
    """
    m1 = microsim.Microsim()
    m2 = microsim.Microsim(random_seed=2.0)
    m3 = microsim.Microsim(random_seed=2.0)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [x.random.random() for x in [m1, m2, m3]]

    assert r1 != r2
    assert r2 == r3


