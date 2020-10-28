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


def test_fit(setup):
    """Test the fitness function"""
    fit = Functions.fit
    assert fit([1,2,3],[1,2,3]) == 0
    # MORE
