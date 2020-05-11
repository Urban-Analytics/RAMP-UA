import pytest
import multiprocessing
import pandas as pd
from microsim import Microsim


@pytest.fixture()
def setup():
    """Used to perform setup stuff that other tests can use"""
    print("\n I'm the fixture - setUp", flush=True)
    yield
    print("\nI'm the fixture - tearDown", flush=True)


def _get_rand(microsim, N=100):
    """Get a random number using the Microsimulation object's random number generator"""
    for _ in range(N):
        microsim.random.random()
    return microsim.random.random()


def test_random(setup):
    """
    Checks that random classes are produce different (or the same!) numbers when they should do
    :return:
    """
    m1 = microsim.Microsim(read_data=False)
    m2 = microsim.Microsim(random_seed=2.0, read_data=False)
    m3 = microsim.Microsim(random_seed=2.0, read_data=False)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [_get_rand(x) for x in [m1, m2, m3]]

    assert r1 != r2
    assert r2 == r3

    # Check that this still happens even if they are executed in pools.
    # Create a large number of microsims and check that all random numbers are unique
    pool = multiprocessing.Pool()
    m = [microsim.Microsim(read_data=False) for _ in range(10000)]
    r = pool.map(_get_rand, m)
    assert len(r) == len(set(r))


def test_extract_msoas_from_indiviuals():
    """Check that a list of areas can be successfully extracted from a DataFrame of indviduals"""
    individuals = pd.DataFrame(data={"Area":["C", "A", "F"]})
    areas = Microsim.extract_msoas_from_indiviuals(individuals)
    assert len(areas) == 3
    # Check the order is correct too
    assert False not in [ x==y for (x,y) in zip(areas,["A", "C", "F"])]
