# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""Comparison of verhulst2018-rs with the original Python/C model (read-only source dir).

Function-level tests compare bit patterns (the port reproduces the original arithmetic on Apple
arm64); end-to-end tests compare short simulations bitwise and also assert the specified
tolerances (linear ≤ 1e-6 relative). Set VERHULST_SOURCE_DIR to point at the original model.
"""

from __future__ import annotations

import ctypes
import platform

import numpy as np
import pytest

import verhulst2018_rs as vr
from reference import MODEL_CONFIG, SOURCE_DIR, load_original, run_original, starting_poles

pytestmark = pytest.mark.skipif(not (SOURCE_DIR / "cochlear_model2018.py").is_file(), reason="original model not found")
BITWISE = platform.machine() in ("arm64", "aarch64")  # FMA patterns reproduced for arm64 builds


@pytest.fixture(scope="module")
def orig():
    return load_original()


@pytest.fixture(scope="module")
def poles():
    return starting_poles()


def click(n=1200, level_pa=0.2):
    s = np.zeros(n + 2)
    s[100:110] = level_pa  # 100 µs rectangular click at 1 ms
    return s


def assert_same(a, b, rel=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    if BITWISE and rel == 0.0:
        np.testing.assert_array_equal(a, b)
    else:
        den = max(np.linalg.norm(a), 1e-300)
        assert np.linalg.norm(a - b) / den <= max(rel, 1e-12)


def init_pair(orig, poles, stim, **cfg):
    """Original model after init_model + solve() prologue, and the Rust core."""
    c = {**MODEL_CONFIG, **cfg}
    m = orig.cochlea_model()
    m.init_model(stim, sheraPo=poles, probe_freq="abr", **c)
    m.Vtmp = np.zeros_like(m.x)
    m.Ytmp = np.zeros_like(m.x)
    m.polecalculation()
    m.SheraParameters()
    m.ZweigImpedance()
    core = vr.make_core(sheraPo=poles, **{k: v for k, v in c.items()})
    return m, core


# ---------------------------------------------------------------- C kernels (cochlea_utils.c)


def test_cubic_interpolate(orig):
    lib = orig.libtrisolv
    f = lib.cubic_interpolate
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_float] * 5
    rng = np.random.default_rng(1)
    for _ in range(5000):
        y = rng.normal(size=4) * 10.0 ** rng.integers(-12, 2)
        mu = rng.random()
        assert f(*y, mu) == vr.cubic_interpolate(*y, mu) or not BITWISE


def test_delay_line(orig):
    lib = orig.libtrisolv
    rng = np.random.default_rng(2)
    n, m = 64, 257
    buf = np.ascontiguousarray(rng.normal(size=(n, m)) * 1e-9)
    z1 = rng.integers(0, m, n).astype(np.int32)
    zr = ((z1 + m - 1) % m).astype(np.int32)
    z2 = ((z1 + 1) % m).astype(np.int32)
    z3 = ((z2 + 1) % m).astype(np.int32)
    dev = np.ascontiguousarray(rng.random(n) * 2.0)  # covers both branches (dev < 1, dev >= 1)
    out = np.zeros(n)
    P = ctypes.POINTER
    lib.delay_line(buf.ctypes.data_as(P(ctypes.c_double)), zr.ctypes.data_as(P(ctypes.c_int)),
                   z1.ctypes.data_as(P(ctypes.c_int)), z2.ctypes.data_as(P(ctypes.c_int)),
                   z3.ctypes.data_as(P(ctypes.c_int)), dev.ctypes.data_as(P(ctypes.c_double)),
                   out.ctypes.data_as(P(ctypes.c_double)), ctypes.c_int(m), ctypes.c_int(n))
    ours = vr.delay_line(buf, list(zr), list(z1), list(z2), list(z3), dev)
    assert_same(out, ours)


def test_solve_tridiagonal(orig, poles):
    m, core = init_pair(orig, poles, click())
    rng = np.random.default_rng(3)
    r = np.ascontiguousarray(rng.normal(size=m.n + 1))
    q = np.zeros_like(r)
    orig.libtrisolv.solve_tridiagonal(ctypes.byref(m.tridata), r.ctypes.data_as(orig.PDOUBLE),
                                      q.ctypes.data_as(orig.PDOUBLE), ctypes.c_int(m.n + 1))
    ours = vr.solve_tridiagonal(m.ZAL, m.ZASC, m.ZAH, r)
    assert_same(q, ours)


# ---------------------------------------------------------------- initialisation


VECTORS = ["x", "f_resonance", "omega", "omega2", "Sherad_factor", "SheraPo", "SheraP", "SheraD", "SheraRho",
           "SheraMu", "PoleS", "PoleE", "Rth", "RthY1", "RthY2", "RthV1", "RthV2", "Sa", "Sb", "const_nl1",
           "cosTheta", "sinTheta", "exact_delay", "ZweigMs", "ZweigMp", "ZASQ", "ZASC", "ZAL", "ZAH", "Dev",
           "Zrp", "Zrp1", "Zrp2", "Zrp3"]
SCALARS = ["dt", "bm_length", "ZweigMso", "ZweigL", "ZweigOmega_co", "ZweigMpo", "Ko", "dx", "q0_factor", "p0x",
           "d_m_factor", "RK4_0", "RK4G_0", "ctr", "onek", "YbufferLgt", "Zwp", "Yknee1", "Yknee2", "Vknee1",
           "Vknee2", "non_linearity"]


@pytest.mark.parametrize("cfg", [
    {},
    {"Zweig_irregularities": 0},
    {"low_freq_irregularities": 0},
    {"non_linearity_type": "disp"},
    {"non_linearity_type": "none"},
    {"subject": 7, "IrrPct": 0.1, "KneeVar": 2.0, "compression_slope": 0.3},
])
def test_init_model_state(orig, poles, cfg):
    m, core = init_pair(orig, poles, click(), **cfg)
    for name in VECTORS:
        assert_same(getattr(m, name), core.vector(name)), name
    for name in SCALARS:
        assert float(getattr(m, name)) == core.scalar(name), name


def test_shera_parameters_and_zweig_impedance(orig, poles):
    m, core = init_pair(orig, poles, click())
    rng = np.random.default_rng(4)
    for zwp in (0, 17, int(m.YbufferLgt) - 1):
        p = 0.02 + 0.29 * rng.random(m.n + 1)
        m.SheraP = p.copy()
        m.Zwp = zwp
        m.SheraParameters()
        m.ZweigImpedance()
        core.set_ybuffer(np.zeros((m.n + 1, int(m.YbufferLgt))), zwp)
        core.set_vector("SheraP", p)
        core.shera_parameters()
        core.zweig_impedance()
        for name in ("SheraD", "SheraRho", "SheraMu", "Dev", "Zrp", "Zrp1", "Zrp2", "Zrp3"):
            assert_same(getattr(m, name), core.vector(name)), name


# ---------------------------------------------------------------- right-hand side (TLsolver)


@pytest.mark.parametrize("amplitude", [1e-12, 3e-6, 3e-4])  # no update / update / saturating poles
def test_tlsolver_single_evaluation(orig, poles, amplitude):
    m, core = init_pair(orig, poles, click())
    n1 = m.n + 1
    rng = np.random.default_rng(5)
    buf = rng.normal(size=(n1, int(m.YbufferLgt))) * 1e-9
    zwp = 123
    m.Ybuffer[:, :] = buf
    m.Zwp = zwp
    m.ZweigImpedance()
    core.set_ybuffer(buf, zwp)
    a, b, c, d = 0.01, -0.02, 0.03, 0.005
    m.interplPoint1, m.interplPoint2, m.interplPoint3, m.interplPoint4 = a, b, c, d
    core.set_interp(a, b, c, d)
    m.lastT = 0.25
    core.set_last_t(0.25)
    for t in (0.25, 0.25 + 0.3e-5, 0.25 + 1e-5):
        y = np.concatenate([rng.normal(size=n1) * amplitude, rng.normal(size=n1) * amplitude * 1e-4])
        d1 = orig.TLsolver(t, y.copy(), m)
        d2 = core.rhs(t, y)
        assert_same(d1, d2)
        for name in ("SheraP", "SheraD", "SheraRho", "Dev", "YZweig", "g", "Qsol", "Zrp1"):
            assert_same(getattr(m, name), core.vector(name)), name


# ---------------------------------------------------------------- end-to-end


def compare_runs(orig, poles, stim, rtol=1e-2, **cfg):
    m, calls = run_original(orig, stim, poles, rtol=rtol, config=cfg, record=True)
    r = vr.simulate(stim, sheraPo=poles, probe_channels=[500], rtol=rtol, **{**MODEL_CONFIG, **cfg})
    return m, calls, r


def test_linear_click(orig, poles):
    stim = click(1500)
    m, calls, r = compare_runs(orig, poles, stim, non_linearity_type="none")
    rel = np.linalg.norm(m.oto_emission - r.oto_emission) / np.linalg.norm(m.oto_emission)
    assert rel <= 1e-6
    assert_same(m.oto_emission, r.oto_emission)
    assert_same(m.Ysolution[:-2, 0], r.displacement[:, 0])
    assert len(calls) == r.stats["rhs_calls"]


@pytest.mark.parametrize("level_pa", [0.02, 0.2, 2.0])
def test_nonlinear_click(orig, poles, level_pa):
    stim = click(1500, level_pa)
    m, calls, r = compare_runs(orig, poles, stim)
    assert_same(m.oto_emission, r.oto_emission)
    assert_same(m.Vsolution[:-2, 0], r.velocity[:, 0])
    assert len(calls) == r.stats["rhs_calls"]


@pytest.mark.parametrize("cfg", [
    {"Zweig_irregularities": 0},
    {"low_freq_irregularities": 0},
    {"non_linearity_type": "disp"},
])
def test_config_variants(orig, poles, cfg):
    stim = click(800, 0.5)
    m, _, r = compare_runs(orig, poles, stim, **cfg)
    assert_same(m.oto_emission, r.oto_emission)


@pytest.mark.parametrize("rtol", [1e-4, 1e-6])
def test_rtol_is_honoured(orig, poles, rtol):
    stim = click(600, 0.2)
    m, calls, r = compare_runs(orig, poles, stim, rtol=rtol)
    assert_same(m.oto_emission, r.oto_emission)
    assert len(calls) == r.stats["rhs_calls"]


def test_tone_with_saturation(orig, poles):
    t = np.arange(2000) / 1e5
    stim = np.concatenate([0.5 * np.sin(2 * np.pi * 2000 * t) * np.minimum(1, t / 5e-3), np.zeros(2)])
    m, calls, r = compare_runs(orig, poles, stim)
    assert_same(m.oto_emission, r.oto_emission)
    assert r.stats["pole_updates"] > 0


def test_accepted_step_variant_runs_and_differs(poles):
    stim = click(1500, 0.2)
    a = vr.simulate(stim, sheraPo=poles, **MODEL_CONFIG)
    b = vr.simulate(stim, sheraPo=poles, pole_update="accepted_step", **MODEL_CONFIG)
    assert np.isfinite(b.oto_emission).all()
    assert not np.array_equal(a.oto_emission, b.oto_emission)
    rel = np.linalg.norm(a.oto_emission - b.oto_emission) / np.linalg.norm(a.oto_emission)
    assert rel < 0.2  # same model, different update timing
    lin_a = vr.simulate(stim, sheraPo=poles, **{**MODEL_CONFIG, "non_linearity_type": "none"})
    lin_b = vr.simulate(stim, sheraPo=poles, pole_update="accepted_step", **{**MODEL_CONFIG, "non_linearity_type": "none"})
    np.testing.assert_array_equal(lin_a.oto_emission, lin_b.oto_emission)  # no poles to update


def test_compact_delay_is_identical(poles):
    stim = click(1200, 0.5)
    a = vr.simulate(stim, sheraPo=poles, **MODEL_CONFIG)
    b = vr.simulate(stim, sheraPo=poles, compact_delay=False, **MODEL_CONFIG)
    np.testing.assert_array_equal(a.oto_emission, b.oto_emission)


def test_probe_freq_semantics(orig, poles):
    stim = click(300, 0.2)
    m = orig.cochlea_model()
    m.init_model(stim, sheraPo=poles, probe_freq="abr", **MODEL_CONFIG)
    r = vr.simulate(stim, sheraPo=poles, probe_freq="abr", **MODEL_CONFIG)
    np.testing.assert_array_equal(r.probe_channels, m.probe_points)
    r = vr.simulate(stim, sheraPo=poles, probe_freq=[1000.0, 4000.0], **MODEL_CONFIG)
    f = m.f_resonance
    assert list(r.probe_channels) == [int(np.argmin(np.abs(f - 1000.0))), int(np.argmin(np.abs(f - 4000.0)))]
