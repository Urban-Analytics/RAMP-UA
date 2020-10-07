import numpy as np

from ramp.params import Params

def test_params_to_from_array():
    params = Params()
    params_array = params.asarray()

    params_from_array = Params.fromarray(params_array)
    params_from_array_array = params_from_array.asarray()

    assert np.all(params_array == params_from_array_array)
