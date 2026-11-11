import numpy as np
import pytest
from helixadd.labels import FAMILY, NESTED, ROLE, assign_label, contains, is_mostly_active, variables
from helixadd.scoring import r2_additive, r2_all, r2_groups, range_r2


def test_example():
    x = np.array([4., 6, 5, 0, 1, 0])
    g = np.array([5, 5, 5, 0, 0, 0])
    assert r2_groups(x, g) == pytest.approx(0.9245, abs=1e-3)


def test_perfect():
    g = np.arange(100) % 10
    x = (g == 5) * 3.0            # fires exactly when the group is 5
    assert r2_groups(x, g) == pytest.approx(1.0)


def test_unrelated():
    g = np.arange(100) % 10
    x = np.tile([1.0, 2.0], 50) * 0 + np.arange(100) // 10   # depends only on the tens
    assert r2_groups(x, g) == pytest.approx(0.0)


def test_r2_all_matches_r2_groups():
    rng = np.random.default_rng(0)
    a = np.repeat(np.arange(100), 100)
    b = np.tile(np.arange(100), 100)
    X = rng.random((10000, 30)).astype(np.float32)
    X[:, 0] = (a + b) % 10            # a perfect last-digit feature
    X[:, 1] = (a // 10) * 0.5         # a perfect magnitude feature
    for name, g in variables(a, b).items():
        fast = r2_all(X, g)
        slow = np.array([r2_groups(X[:, f].astype(np.float64), g) for f in range(X.shape[1])])
        assert fast == pytest.approx(slow, abs=1e-4), name


def test_range_r2_matches_r2_groups():
    rng = np.random.default_rng(1)
    a = np.repeat(np.arange(100), 100)
    X = rng.random((10000, 5)).astype(np.float32)
    X[:, 0] = ((a >= 54) & (a < 64)) * 2.0          # "a around 59": the range [54, 64)
    best, lo, L = range_r2(X, a)
    assert (lo[0], L[0]) == (54, 10)
    assert best[0] == pytest.approx(1.0, abs=1e-3)      # float32 rounding
    for f in range(5):
        inside = ((a >= lo[f]) & (a < lo[f] + L[f])).astype(int)
        assert best[f] == pytest.approx(r2_groups(X[:, f].astype(np.float64), inside), abs=1e-4)


def test_nested_is_complete():
    """NESTED must list exactly the pairs where one variable contains another."""
    a = np.repeat(np.arange(100), 100)
    b = np.tile(np.arange(100), 100)
    g = variables(a, b)
    found = {A: sorted(B for B in g if B != A and contains(g[A], g[B])) for A in g}
    found = {A: Bs for A, Bs in found.items() if Bs}
    assert found == {A: sorted(Bs) for A, Bs in NESTED.items()}


def test_every_variable_has_family_and_role():
    assert set(ROLE) == set(FAMILY)


def test_assign_label_order():
    assert assign_label(0.4, 0.5, "sum") == "unclear"
    assert assign_label(0.9, 0.1, "sum") == "mixed"
    assert assign_label(0.9, 0.5, "sum") == "sum"


def test_mostly_active():
    assert is_mostly_active(9000, 9992)
    assert not is_mostly_active(4000, 9992)


def test_r2_additive():
    a = np.repeat(np.arange(100), 100)
    b = np.tile(np.arange(100), 100)
    A, B = a // 10, b // 10
    tens = A * 10 + B
    X = np.stack([
        0.3 * A + 0.5 * B,                    # two separate effects that add up
        ((A == 5) & (B == 6)) * 2.0,          # one blob: fires only for a in 50s AND b in 60s
    ], axis=1).astype(np.float32)
    add = r2_additive(X, A, B)
    joint = r2_all(X, tens)
    assert add[0] == pytest.approx(1.0, abs=1e-3)      # additive feature: fully explained
    assert joint[1] == pytest.approx(1.0, abs=1e-3)    # the blob: the joint model fits it
    assert add[1] < 0.5                                # ...the additive model does not
