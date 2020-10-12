import numpy as np
import scipy.stats as stats


def rand_exp(nums):
    return -np.log(1.0 - nums)


def rand_weibull(nums, scale, shape):
    return scale * np.power(rand_exp(nums), 1.0 / shape)


# These tests verify that the exponential and weibull samplers in
# the kernels produce the summary statistics expected when they
# are not rounded to integers.
def test_exp_is_exp():
    n = 100000
    expected_rate = 1.0
    exps = rand_exp(np.random.rand(n))
    _, rate = stats.expon.fit(exps, floc=0.0)

    assert np.isclose(expected_rate, rate, atol=0.1)


def test_weibull_is_weibull():
    n = 100000
    expected_scale = 14.0
    expected_shape = 5.0
    weibulls = rand_weibull(np.random.rand(n), expected_scale, expected_shape)
    shape, _, scale = stats.weibull_min.fit(weibulls, floc=0.0)

    assert np.isclose(expected_scale, scale, atol=0.1)
    assert np.isclose(expected_shape, shape, atol=0.1)


def test_exposed_dist():
    n = 100000
    weibulls = rand_weibull(np.random.rand(n), 2.82, 3.93)
    mean = weibulls.mean()
    std = weibulls.std()

    expected_mean = 2.56
    expected_std = 0.72

    assert np.isclose(expected_mean, mean, atol=0.1)
    assert np.isclose(expected_std, std, atol=0.1)


def test_presymptomatic_dist():
    n = 100000
    weibulls = rand_weibull(np.random.rand(n), 2.45, 7.12)
    mean = weibulls.mean()
    std = weibulls.std()

    expected_mean = 2.3
    expected_std = 0.35

    assert np.isclose(expected_mean, mean, atol=0.1)
    assert np.isclose(expected_std, std, atol=0.1)
