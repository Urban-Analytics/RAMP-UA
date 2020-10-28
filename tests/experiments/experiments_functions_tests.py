import pytest
from experiments.experiments_functions import Functions

# ********************************************************
# Tests of the functions in functions.py
# ********************************************************

# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it,
# like `test_step()`.
@pytest.fixture()
def setup():
    # Nothing to set up. Could create dummy data or whatever
    #data = [1,2,3]
    #yield data
    pass


def test_fit_l2(setup):
    """Test the fitness function"""
    f = Functions.fit_l2

    assert f([1, 2, 3],[1, 2, 3]) == 0
    assert f([1, 2, 3],[1, 2, 4]) == 1
    assert f([1, 2, 3],[1, 2, 5]) == 2

    with pytest.raises(Exception):
        f([1, 2, 3], [1, 2])
        f(1, [1, 2])
        f("b", "a")
