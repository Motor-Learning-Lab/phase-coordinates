"""
Tests for phase_coordinates.core
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from phase_coordinates import hilbert_phase, cycle_by_cycle_pca_coordinates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cyclic_3d(
    n_cycles=5,
    samples_per_cycle=100,
    noise_std=0.05,
    radius=1.0,
    tilt_angle=np.pi / 6,
    rng=None,
):
    """
    Generate synthetic 3D noisy cyclic data.

    The "true" motion is a circle of given *radius* lying in a plane that is
    tilted by *tilt_angle* around the x-axis. Gaussian noise is added in all
    three dimensions.

    Returns
    -------
    X : ndarray, shape (n_time, 3)
    phase_true : ndarray, shape (n_time,) - unwrapped true phase
    fs : float - sampling rate (Hz)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_time = n_cycles * samples_per_cycle
    fs = float(samples_per_cycle)  # 1 cycle per second

    t = np.arange(n_time) / fs
    phase_true = 2 * np.pi * t  # one cycle per second, unwrapped

    # Circle in the tilted plane
    u = radius * np.cos(phase_true)
    v = radius * np.sin(phase_true)

    # Rotate: x = u, y = v*cos(tilt), z = v*sin(tilt)
    x = u
    y = v * np.cos(tilt_angle)
    z = v * np.sin(tilt_angle)

    X = np.column_stack([x, y, z])
    X += rng.standard_normal(X.shape) * noise_std

    return X, phase_true, fs


def _rotation_matrix_x(angle):
    """Rotation matrix around the x-axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_matrix_z(angle):
    """Rotation matrix around the z-axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _make_changing_planes(
    n_cycles=6,
    samples_per_cycle=120,
    noise_std=0.01,
    rng=None,
):
    """
    Generate 3D cyclic data where each cycle lies in a *different* plane.

    The local motion in cycle *k* is a unit circle in the (u, v) plane.
    The plane is tilted by a different angle for each cycle via a rotation
    around the z-axis.

    Returns
    -------
    X          : ndarray (n_time, 3)
    phase_true : ndarray (n_time,)  - unwrapped true phase
    true_normals : list of ndarray (3,) - true plane normal for each cycle
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_time = n_cycles * samples_per_cycle
    t = np.arange(n_time) / float(samples_per_cycle)
    phase_true = 2 * np.pi * t

    X = np.zeros((n_time, 3))
    true_normals = []

    for cyc in range(n_cycles):
        # Different tilt for every cycle: cycle k tilted by k * 30 degrees
        tilt = cyc * (np.pi / 6)  # 30 degrees per cycle
        R = _rotation_matrix_z(tilt)

        # True normal of this cycle's plane (z-axis rotated by R)
        normal = R @ np.array([0.0, 0.0, 1.0])
        true_normals.append(normal)

        start = cyc * samples_per_cycle
        end = start + samples_per_cycle
        theta = phase_true[start:end] - phase_true[start]

        # Unit circle in local (u, v, 0) then rotate
        local = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))])
        X[start:end] = (R @ local.T).T

    X += rng.standard_normal(X.shape) * noise_std

    return X, phase_true, true_normals


# ---------------------------------------------------------------------------
# hilbert_phase
# ---------------------------------------------------------------------------

class TestHilbertPhase:
    """Tests for hilbert_phase."""

    def test_returns_three_arrays(self):
        rng = np.random.default_rng(0)
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t) + rng.standard_normal(200) * 0.05
        result = hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))
        assert len(result) == 3

    def test_output_shapes(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        unwrapped, wrapped, amp = hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))
        assert unwrapped.shape == (200,)
        assert wrapped.shape == (200,)
        assert amp.shape == (200,)

    def test_wrapped_phase_in_range(self):
        fs = 100.0
        t = np.arange(500) / fs
        sig = np.sin(2 * np.pi * 3 * t)
        _, wrapped, _ = hilbert_phase(sig, fs=fs, f_range=(2.0, 5.0))
        assert np.all(wrapped >= -np.pi - 1e-10)
        assert np.all(wrapped <= np.pi + 1e-10)

    def test_unwrapped_phase_monotone(self):
        """Unwrapped phase should be roughly monotonically increasing."""
        fs = 200.0
        t = np.arange(1000) / fs
        sig = np.sin(2 * np.pi * 5 * t)
        unwrapped, _, _ = hilbert_phase(sig, fs=fs, f_range=(3.0, 8.0))
        # Ignore transient edges: phase differences in the middle should be >= 0
        diff = np.diff(unwrapped[100:-100])
        assert np.sum(diff < -0.1) == 0, "Unwrapped phase has large decreases"

    def test_phase_rate_matches_frequency(self):
        """
        For a pure sinusoid at f Hz, the mean rate of phase increase should be
        approximately 2pi*f rad/s.
        """
        fs = 500.0
        f = 3.0
        t = np.arange(2000) / fs
        sig = np.sin(2 * np.pi * f * t)
        unwrapped, _, _ = hilbert_phase(sig, fs=fs, f_range=(1.5, 6.0))
        # Skip edges affected by filter transients; 0.5 rad/s tolerance is well
        # within the expected accuracy of the Hilbert phase estimate for a clean
        # sinusoid after trimming 10% from each end of the 2000-sample signal.
        rate = np.mean(np.diff(unwrapped[200:-200])) * fs  # rad/s
        assert abs(rate - 2 * np.pi * f) < 0.5, f"Phase rate {rate:.3f} far from 2pi*{f}"

    def test_amplitude_positive(self):
        fs = 100.0
        t = np.arange(300) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        _, _, amp = hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))
        assert np.all(amp >= 0)

    def test_accepts_list_input(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = list(np.sin(2 * np.pi * 2 * t))
        unwrapped, wrapped, amp = hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))
        assert unwrapped.shape == (200,)

    # -- input validation --

    def test_raises_on_2d_input(self):
        fs = 100.0
        sig = np.ones((50, 2))
        with pytest.raises(ValueError, match="1-D"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))

    def test_raises_on_nan_input(self):
        fs = 100.0
        sig = np.sin(np.linspace(0, 4 * np.pi, 100))
        sig[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))

    def test_raises_on_inf_input(self):
        fs = 100.0
        sig = np.sin(np.linspace(0, 4 * np.pi, 100)).copy()
        sig[5] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))

    def test_raises_on_signal_too_short(self):
        fs = 100.0
        sig = np.ones(5)  # fewer than 13 samples
        with pytest.raises(ValueError, match="too short"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 4.0))

    def test_raises_on_non_positive_fs(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        with pytest.raises(ValueError, match="positive"):
            hilbert_phase(sig, fs=0.0, f_range=(1.0, 4.0))
        with pytest.raises(ValueError, match="positive"):
            hilbert_phase(sig, fs=-10.0, f_range=(1.0, 4.0))

    def test_raises_on_invalid_f_range_low_ge_high(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        with pytest.raises(ValueError, match="low < high"):
            hilbert_phase(sig, fs=fs, f_range=(5.0, 2.0))
        with pytest.raises(ValueError, match="low < high"):
            hilbert_phase(sig, fs=fs, f_range=(3.0, 3.0))

    def test_raises_on_f_range_above_nyquist(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        with pytest.raises(ValueError, match="Nyquist"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 50.0))  # high == Nyquist
        with pytest.raises(ValueError, match="Nyquist"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 60.0))  # high > Nyquist

    def test_raises_on_f_range_wrong_length(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        with pytest.raises(ValueError, match="length-2"):
            hilbert_phase(sig, fs=fs, f_range=(1.0,))
        with pytest.raises(ValueError, match="length-2"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 2.0, 3.0))

    def test_raises_on_f_range_low_not_positive(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.sin(2 * np.pi * 2 * t)
        with pytest.raises(ValueError, match="0 < low < high"):
            hilbert_phase(sig, fs=fs, f_range=(0.0, 10.0))

    # -- non-monotonic phase warning --

    def test_warns_on_non_monotonic_phase(self):
        """
        A sum of two close frequencies creates beats (amplitude nulls).
        At each null the Hilbert phase jumps, producing large negative steps.
        This should trigger a UserWarning about unreliable phase.
        """
        fs = 200.0
        t = np.arange(2000) / fs
        # f1 and f2 both inside the bandpass; beat frequency = |f1-f2| = 1 Hz
        # creates amplitude nulls ~10 times in 10 s, each causing a pi phase jump
        f1, f2 = 1.5, 2.5
        sig = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
        with pytest.warns(UserWarning, match="negative steps"):
            hilbert_phase(sig, fs=fs, f_range=(1.0, 3.5))


# ---------------------------------------------------------------------------
# cycle_by_cycle_pca_coordinates
# ---------------------------------------------------------------------------

class TestCycleByCyclePcaCoordinates:
    """Tests for cycle_by_cycle_pca_coordinates."""

    # -- basic smoke tests --

    def test_returns_dataframe_and_dict(self):
        X, phase_true, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        assert isinstance(coords, pd.DataFrame)
        assert isinstance(models, dict)

    def test_output_columns(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        expected = {
            "cycle", "phase", "phase_wrapped", "phase_in_cycle", "amp_hilbert",
            "pc1_local", "pc2_local", "pc3_local",
            "theta_local", "theta_local_wrapped",
            "radius_local", "perp_local",
        }
        assert expected.issubset(set(coords.columns))

    def test_output_length(self):
        n_time = 500
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        assert len(coords) == len(X)

    # -- input variants --

    def test_dataframe_input(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=4, samples_per_cycle=80)
        df = pd.DataFrame(X, columns=["x", "y", "z"])
        coords, models = cycle_by_cycle_pca_coordinates(df, phase=phase_true)
        assert len(coords) == len(df)

    def test_dataframe_with_columns_kwarg(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=4, samples_per_cycle=80)
        df = pd.DataFrame(
            np.hstack([X, np.ones((len(X), 1))]),
            columns=["x", "y", "z", "extra"],
        )
        coords, models = cycle_by_cycle_pca_coordinates(
            df, phase=phase_true, columns=["x", "y", "z"]
        )
        assert len(coords) == len(df)

    def test_with_ref_signal_and_fs(self):
        X, _, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100, noise_std=0.01)
        ref = X[:, 0]  # use x-coordinate as reference signal
        coords, models = cycle_by_cycle_pca_coordinates(
            X, ref_signal=ref, fs=fs, f_range=(0.5, 2.0)
        )
        assert len(coords) == len(X)
        assert len(models) >= 4  # should detect most cycles

    # -- error conditions --

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="shape"):
            cycle_by_cycle_pca_coordinates(
                np.arange(100, dtype=float),
                phase=np.linspace(0, 10 * np.pi, 100),
            )

    def test_raises_on_fewer_than_3_features(self):
        X = np.random.default_rng(0).standard_normal((100, 2))
        with pytest.raises(ValueError, match="3 features"):
            cycle_by_cycle_pca_coordinates(
                X, phase=np.linspace(0, 10 * np.pi, 100)
            )

    def test_raises_without_phase_info(self):
        X, _, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=80)
        with pytest.raises(ValueError, match="phase"):
            cycle_by_cycle_pca_coordinates(X)

    def test_raises_on_phase_length_mismatch(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=80)
        with pytest.raises(ValueError, match="length"):
            cycle_by_cycle_pca_coordinates(X, phase=phase_true[:-10])

    # -- cycle detection --

    def test_correct_number_of_cycles(self):
        n_cycles = 6
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=n_cycles, samples_per_cycle=100, noise_std=0.02
        )
        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        # cycle ids should span from 0 to n_cycles-1 (approximately)
        cycle_ids = coords["cycle"].unique()
        assert len(cycle_ids) <= n_cycles + 1
        assert len(models) >= n_cycles - 1  # allow one partial cycle at edges

    def test_phase_in_cycle_range(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        assert coords["phase_in_cycle"].min() >= 0
        assert coords["phase_in_cycle"].max() < 2 * np.pi + 1e-10

    def test_amp_hilbert_nan_when_phase_provided(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        assert coords["amp_hilbert"].isna().all()

    def test_amp_hilbert_finite_when_ref_signal_provided(self):
        X, _, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100, noise_std=0.01)
        coords, _ = cycle_by_cycle_pca_coordinates(
            X, ref_signal=X[:, 0], fs=fs, f_range=(0.5, 2.0)
        )
        assert coords["amp_hilbert"].notna().any()

    # -- geometric validity --

    def test_radius_non_negative(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        valid = coords["radius_local"].dropna()
        assert (valid >= 0).all()

    def test_radius_equals_hypot_of_pc1_pc2(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        mask = coords["radius_local"].notna()
        r_computed = np.hypot(coords.loc[mask, "pc1_local"], coords.loc[mask, "pc2_local"])
        np.testing.assert_allclose(coords.loc[mask, "radius_local"].to_numpy(), r_computed, atol=1e-12)

    def test_theta_wrapped_in_range(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        valid = coords["theta_local_wrapped"].dropna()
        assert (valid >= -np.pi - 1e-10).all()
        assert (valid <= np.pi + 1e-10).all()

    def test_perp_local_equals_pc3(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        mask = coords["perp_local"].notna()
        np.testing.assert_allclose(
            coords.loc[mask, "perp_local"].to_numpy(),
            coords.loc[mask, "pc3_local"].to_numpy(),
            atol=1e-12,
        )

    # -- data recovery --

    def test_data_recovery(self):
        """
        Reconstruct X from the per-cycle PCA models using iloc (positional
        indexing) and verify it matches the original data to within numerical
        precision.
        """
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=100, noise_std=0.0
        )
        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)

        X_rec = np.full_like(X, np.nan)
        for cyc, model in models.items():
            idx = model["indices"]  # positional integer indices
            center = model["center"]
            comps = model["components"]  # (3, n_features)
            p1 = coords.iloc[idx]["pc1_local"].to_numpy()
            p2 = coords.iloc[idx]["pc2_local"].to_numpy()
            p3 = coords.iloc[idx]["pc3_local"].to_numpy()
            X_rec[idx] = (
                p1[:, None] * comps[0]
                + p2[:, None] * comps[1]
                + p3[:, None] * comps[2]
                + center
            )

        valid = ~np.isnan(X_rec).any(axis=1)
        np.testing.assert_allclose(X[valid], X_rec[valid], atol=1e-10)

    def test_data_recovery_non_default_index(self):
        """
        Reconstruction via iloc works correctly even when the DataFrame has a
        non-default (e.g. time-based) index.
        """
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=100, noise_std=0.0
        )
        n_time = len(X)
        # Non-default float time index
        time_index = pd.Index(np.arange(n_time) / 100.0, name="time_s")
        df = pd.DataFrame(X, columns=["x", "y", "z"], index=time_index)

        coords, models = cycle_by_cycle_pca_coordinates(df, phase=phase_true)

        # The coords DataFrame inherits the non-default index.
        assert (coords.index == time_index).all()

        X_rec = np.full_like(X, np.nan)
        for cyc, model in models.items():
            idx = model["indices"]  # always positional ints
            center = model["center"]
            comps = model["components"]
            # Must use iloc for positional access, not loc
            p1 = coords.iloc[idx]["pc1_local"].to_numpy()
            p2 = coords.iloc[idx]["pc2_local"].to_numpy()
            p3 = coords.iloc[idx]["pc3_local"].to_numpy()
            X_rec[idx] = (
                p1[:, None] * comps[0]
                + p2[:, None] * comps[1]
                + p3[:, None] * comps[2]
                + center
            )

        valid = ~np.isnan(X_rec).any(axis=1)
        np.testing.assert_allclose(X[valid], X_rec[valid], atol=1e-10)

    def test_data_recovery_approximate_for_high_dimensional_input(self):
        """
        For input with more than 3 features, reconstruction from only 3 PCs is
        approximate. Verify that the error is non-trivial when the data has
        significant variance beyond the first 3 components.
        """
        rng = np.random.default_rng(99)
        n_time = 600
        phase_true = np.linspace(0, 10 * np.pi, n_time)
        # 5D data where all 5 dimensions carry independent signal
        X = rng.standard_normal((n_time, 5))

        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)

        X_rec = np.full_like(X, np.nan)
        for cyc, model in models.items():
            idx = model["indices"]
            center = model["center"]
            comps = model["components"]  # shape (3, 5)
            p1 = coords.iloc[idx]["pc1_local"].to_numpy()
            p2 = coords.iloc[idx]["pc2_local"].to_numpy()
            p3 = coords.iloc[idx]["pc3_local"].to_numpy()
            X_rec[idx] = (
                p1[:, None] * comps[0]
                + p2[:, None] * comps[1]
                + p3[:, None] * comps[2]
                + center
            )

        valid = ~np.isnan(X_rec).any(axis=1)
        max_error = np.abs(X[valid] - X_rec[valid]).max()
        # Reconstruction from 3 of 5 PCs is lossy for isotropic noise data.
        assert max_error > 1e-6, (
            f"Expected approximate (lossy) reconstruction for 5D data, "
            f"but max error was only {max_error:.2e}"
        )

    def test_radius_close_to_true_radius(self):
        """
        For near-noiseless circular data, the median radius should be close to
        the true radius.
        """
        true_radius = 2.0
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=200, noise_std=0.01, radius=true_radius
        )
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        median_radius = coords["radius_local"].median()
        assert abs(median_radius - true_radius) < 0.1, (
            f"Median radius {median_radius:.3f} far from true {true_radius}"
        )

    def test_perp_small_for_planar_data(self):
        """
        For data that lies nearly in a plane, perpendicular deviation should be
        small relative to the in-plane radius.
        """
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=200, noise_std=0.02
        )
        coords, _ = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        rms_perp = np.sqrt((coords["perp_local"].dropna() ** 2).mean())
        rms_radius = np.sqrt((coords["radius_local"].dropna() ** 2).mean())
        assert rms_perp < 0.5 * rms_radius, (
            f"perp RMS {rms_perp:.4f} not small relative to radius RMS {rms_radius:.4f}"
        )

    # -- changing planes --

    def test_local_pca_plane_tracks_true_plane(self):
        """
        When each cycle lies in a different plane, the fitted local PCA normal
        for that cycle should be close to the true plane normal.

        The plane normal is components[2] (the PC3 axis = the axis with
        smallest variance, i.e. perpendicular to the plane). Alignment is
        checked up to sign: abs(dot(fitted_normal, true_normal)) ~ 1.
        """
        X, phase_true, true_normals = _make_changing_planes(
            n_cycles=6, samples_per_cycle=120, noise_std=0.01
        )
        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)

        cycle_ids = sorted(models.keys())
        for cyc in cycle_ids:
            m = models[cyc]
            fitted_normal = m["components"][2]  # smallest-variance direction
            true_normal = true_normals[cyc]
            alignment = abs(np.dot(fitted_normal, true_normal))
            assert alignment > 0.97, (
                f"Cycle {cyc}: fitted plane normal alignment with true normal "
                f"is {alignment:.4f} (expected > 0.97)"
            )

    def test_perp_small_for_each_cycle_in_changing_planes(self):
        """
        Even when each cycle lies in a different plane, perp_local should be
        small for near-planar cycle data.
        """
        X, phase_true, _ = _make_changing_planes(
            n_cycles=6, samples_per_cycle=120, noise_std=0.01
        )
        coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)

        for cyc, m in models.items():
            idx = m["indices"]
            perp = coords.iloc[idx]["perp_local"].to_numpy()
            rms_perp = np.sqrt(np.mean(perp ** 2))
            assert rms_perp < 0.1, (
                f"Cycle {cyc}: perp RMS {rms_perp:.4f} too large for near-planar data"
            )

    # -- models dict contents --

    def test_models_keys(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        _, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        for cyc, m in models.items():
            assert isinstance(cyc, int)
            assert "pca" in m
            assert "center" in m
            assert "components" in m
            assert "explained_variance_ratio" in m
            assert "indices" in m

    def test_models_components_shape(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        _, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        for m in models.values():
            assert m["components"].shape == (3, 3)

    def test_models_explained_variance_sums_to_at_most_one(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        _, models = cycle_by_cycle_pca_coordinates(X, phase=phase_true)
        for m in models.values():
            total_var = m["explained_variance_ratio"].sum()
            assert total_var <= 1.0 + 1e-10

    # -- min_samples_per_cycle --

    def test_min_samples_per_cycle_skips_short_cycles(self):
        """
        If min_samples_per_cycle is very large, all cycles should be skipped
        and models should be empty.
        """
        X, phase_true, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=50)
        _, models = cycle_by_cycle_pca_coordinates(
            X, phase=phase_true, min_samples_per_cycle=1000
        )
        assert len(models) == 0
