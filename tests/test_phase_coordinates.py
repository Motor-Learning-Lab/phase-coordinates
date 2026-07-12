"""
Tests for phase_coordinates.core
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from phase_coordinates import (
    hilbert_phase,
    fit_pca_phase_coordinates,
    reconstruct_phase_coordinates,
    identify_cycles_from_phase,
    SAMPLE_COLUMNS,
    CYCLE_COLUMNS,
)


def _epochs(phase, fs):
    """Helper: build CycleEpochs from a phase array."""
    return identify_cycles_from_phase(np.asarray(phase, dtype=float), sampling_rate_hz=fs)


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
    The plane is tilted by a changing x-rotation (tilt) and optionally yawed
    by a z-rotation; it is the x-rotation that moves the plane normal away
    from ``[0, 0, 1]``, while the z-rotation yaws the already-tilted plane.

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
        # Different yaw and tilt for every cycle so the plane normal actually
        # changes: tilt rotates the z-axis away from [0,0,1], producing a
        # normal with varying x/y/z components.
        yaw = cyc * (np.pi / 6)   # 30 degrees per cycle (optional for normal)
        tilt = cyc * (np.pi / 6)  # 30 degrees per cycle — this changes normal
        R = _rotation_matrix_z(yaw) @ _rotation_matrix_x(tilt)

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

    def test_raises_on_signal_too_short_for_sosfiltfilt(self):
        """
        A signal > _HILBERT_MIN_SAMPLES but still too short for sosfiltfilt
        (padlen=27 for 4-section SOS) should raise a clear ValueError, not a
        raw scipy padding error.
        """
        fs = 100.0
        # 20 samples: passes the _HILBERT_MIN_SAMPLES=13 check but is shorter
        # than sosfiltfilt's padlen of 27 for a 4th-order bandpass SOS filter.
        sig = np.sin(np.linspace(0, 2 * np.pi, 20))
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
# TestFitPcaPhaseCoordinates
# ---------------------------------------------------------------------------

class TestFitPcaPhaseCoordinates:
    """Tests for fit_pca_phase_coordinates (new public API)."""

    # -- basic smoke tests --

    def test_returns_three_tuple(self):
        X, phase_true, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        result = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert len(result) == 3

    def test_returns_dataframe_dict_tuple(self):
        X, phase_true, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert isinstance(samples, pd.DataFrame)
        assert isinstance(cycles, pd.DataFrame)
        assert isinstance(details, dict)

    def test_sample_columns(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert list(samples.columns) == SAMPLE_COLUMNS

    def test_cycle_columns(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert list(cycles.columns) == CYCLE_COLUMNS

    def test_output_length(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert len(samples) == len(X)

    def test_details_algorithm_key(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        _, _, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert details["algorithm"] == "pca"
        assert "models" in details

    # -- input variants --

    def test_dataframe_input(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=4, samples_per_cycle=80)
        df = pd.DataFrame(X, columns=["x", "y", "z"])
        samples, cycles, details = fit_pca_phase_coordinates(df, epochs=_epochs(phase_true, 100.0))
        assert len(samples) == len(df)

    def test_dataframe_with_columns_kwarg(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=4, samples_per_cycle=80)
        df = pd.DataFrame(
            np.hstack([X, np.ones((len(X), 1))]),
            columns=["x", "y", "z", "extra"],
        )
        samples, cycles, details = fit_pca_phase_coordinates(
            df, epochs=_epochs(phase_true, 100.0), columns=["x", "y", "z"]
        )
        assert len(samples) == len(df)

    def test_with_ref_signal_and_fs(self):
        # In the new pipeline, phase estimation is a separate stage.  Build
        # the epochs from the Hilbert phase and confirm downstream fitting.
        X, _, fs = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100, noise_std=0.01)
        ref = X[:, 0]
        phase_h, _, _ = hilbert_phase(ref, fs=fs, f_range=(0.5, 2.0))
        samples, cycles, details = fit_pca_phase_coordinates(
            X, epochs=_epochs(phase_h, fs)
        )
        assert len(samples) == len(X)
        assert len(cycles) >= 4  # should detect most cycles

    # -- error conditions --

    def test_raises_on_1d_input(self):
        phase = np.linspace(0, 10 * np.pi, 100)
        with pytest.raises(ValueError, match="shape"):
            fit_pca_phase_coordinates(
                np.arange(100, dtype=float),
                epochs=_epochs(phase, 100.0),
            )

    def test_raises_on_fewer_than_3_features(self):
        X = np.random.default_rng(0).standard_normal((100, 2))
        phase = np.linspace(0, 10 * np.pi, 100)
        with pytest.raises(ValueError, match="3 features"):
            fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))

    def test_raises_without_epochs(self):
        X, _, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=80)
        with pytest.raises(TypeError):
            fit_pca_phase_coordinates(X)

    def test_raises_on_epochs_length_mismatch(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=80)
        # Build epochs from a truncated phase so cycle_index length != len(X).
        short_epochs = _epochs(phase_true[:-10], 100.0)
        with pytest.raises(ValueError, match="length"):
            fit_pca_phase_coordinates(X, epochs=short_epochs)

    def test_raises_on_2d_phase(self):
        phase_true = np.linspace(0, 6 * np.pi, 240)
        phase_2d = np.tile(phase_true, (2, 1))  # shape (2, n_time)
        with pytest.raises(ValueError, match="1-D"):
            _epochs(phase_2d, 100.0)

    def test_raises_on_phase_with_nan(self):
        phase_nan = np.linspace(0, 6 * np.pi, 240)
        phase_nan[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _epochs(phase_nan, 100.0)

    def test_raises_on_phase_with_inf(self):
        phase_inf = np.linspace(0, 6 * np.pi, 240)
        phase_inf[50] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _epochs(phase_inf, 100.0)

    # -- cycle detection --

    def test_correct_number_of_cycles(self):
        n_cycles = 6
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=n_cycles, samples_per_cycle=100, noise_std=0.02
        )
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        # cycle ids should span from 0 to n_cycles-1 (approximately)
        cycle_ids = samples["cycle"].unique()
        assert len(cycle_ids) <= n_cycles + 1
        assert len(cycles) >= n_cycles - 1  # allow one partial cycle at edges

    def test_phase_in_cycle_range(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert samples["phase_in_cycle"].min() >= 0
        assert samples["phase_in_cycle"].max() < 2 * np.pi + 1e-10

    def test_details_carries_epochs_provenance(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        _, _, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        assert details["epochs_source"] == "phase"
        assert "sampling_rate_hz" in details["epochs_metadata"]

    # -- geometric validity --

    def test_radius_non_negative(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        valid = samples["radius"].dropna()
        assert (valid >= 0).all()

    def test_radius_equals_hypot_of_u_v(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        mask = samples["radius"].notna()
        r_computed = np.hypot(samples.loc[mask, "u"], samples.loc[mask, "v"])
        np.testing.assert_allclose(samples.loc[mask, "radius"].to_numpy(), r_computed, atol=1e-12)

    def test_theta_wrapped_in_range(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        valid = samples["theta_wrapped"].dropna()
        assert (valid >= -np.pi - 1e-10).all()
        assert (valid <= np.pi + 1e-10).all()

    # -- data recovery --

    def test_data_recovery_via_reconstruct(self):
        """
        Reconstruct X via reconstruct_phase_coordinates and verify near-exact
        recovery for zero-noise 3-D data.
        """
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=100, noise_std=0.0
        )
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        X_hat = reconstruct_phase_coordinates(samples, cycles)
        valid = ~np.isnan(X_hat[:, 0])
        assert valid.sum() > 0
        np.testing.assert_allclose(X[valid], X_hat[valid], atol=1e-10)

    def test_data_recovery_non_default_index(self):
        """
        Reconstruction works correctly even when the DataFrame has a
        non-default (e.g. time-based) index.
        """
        X, phase_true, _ = _make_cyclic_3d(
            n_cycles=5, samples_per_cycle=100, noise_std=0.0
        )
        n_time = len(X)
        time_index = pd.Index(np.arange(n_time) / 100.0, name="time_s")
        df = pd.DataFrame(X, columns=["x", "y", "z"], index=time_index)

        samples, cycles, details = fit_pca_phase_coordinates(df, epochs=_epochs(phase_true, 100.0))

        # The samples DataFrame inherits the non-default index.
        assert (samples.index == time_index).all()

        X_hat = reconstruct_phase_coordinates(samples, cycles)
        valid = ~np.isnan(X_hat[:, 0])
        np.testing.assert_allclose(X[valid], X_hat[valid], atol=1e-10)

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

        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]

        X_rec = np.full_like(X, np.nan)
        for cyc, model in models.items():
            idx = model["indices"]
            center = model["center"]
            comps = model["components"]  # shape (3, 5)
            p1 = samples.iloc[idx]["u"].to_numpy()
            p2 = samples.iloc[idx]["v"].to_numpy()
            p3 = samples.iloc[idx]["perp"].to_numpy()
            X_rec[idx] = (
                p1[:, None] * comps[0]
                + p2[:, None] * comps[1]
                + p3[:, None] * comps[2]
                + center
            )

        valid = ~np.isnan(X_rec).any(axis=1)
        max_error = np.abs(X[valid] - X_rec[valid]).max()
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
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        median_radius = samples["radius"].median()
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
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        rms_perp = np.sqrt((samples["perp"].dropna() ** 2).mean())
        rms_radius = np.sqrt((samples["radius"].dropna() ** 2).mean())
        assert rms_perp < 0.5 * rms_radius, (
            f"perp RMS {rms_perp:.4f} not small relative to radius RMS {rms_radius:.4f}"
        )

    # -- changing planes --

    def test_changing_planes_normals_actually_vary(self):
        """
        _make_changing_planes must produce true normals that genuinely differ
        across cycles.
        """
        _, _, true_normals = _make_changing_planes(n_cycles=6, samples_per_cycle=120)
        normals = np.array(true_normals)
        assert np.std(normals, axis=0).max() > 0.1, (
            "True normals from _make_changing_planes do not vary across cycles "
            f"(max per-axis std = {np.std(normals, axis=0).max():.4f})."
        )

    def test_local_pca_plane_tracks_true_plane(self):
        """
        When each cycle lies in a different plane, the fitted local PCA normal
        for that cycle should be close to the true plane normal.
        """
        X, phase_true, true_normals = _make_changing_planes(
            n_cycles=6, samples_per_cycle=120, noise_std=0.01
        )
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]

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
        Even when each cycle lies in a different plane, perp should be
        small for near-planar cycle data.
        """
        X, phase_true, _ = _make_changing_planes(
            n_cycles=6, samples_per_cycle=120, noise_std=0.01
        )
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]

        for cyc, m in models.items():
            idx = m["indices"]
            perp = samples.iloc[idx]["perp"].to_numpy()
            rms_perp = np.sqrt(np.mean(perp ** 2))
            assert rms_perp < 0.1, (
                f"Cycle {cyc}: perp RMS {rms_perp:.4f} too large for near-planar data"
            )

    # -- models dict contents --

    def test_models_keys(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]
        for cyc, m in models.items():
            assert isinstance(cyc, int)
            assert "pca" in m
            assert "center" in m
            assert "components" in m
            assert "explained_variance_ratio" in m
            assert "indices" in m

    def test_models_components_shape(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]
        for m in models.values():
            assert m["components"].shape == (3, 3)

    def test_models_explained_variance_sums_to_at_most_one(self):
        X, phase_true, _ = _make_cyclic_3d(n_cycles=5, samples_per_cycle=100)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase_true, 100.0))
        models = details["models"]
        for m in models.values():
            total_var = m["explained_variance_ratio"].sum()
            assert total_var <= 1.0 + 1e-10

    # -- min_samples_per_cycle --

    def test_min_samples_per_cycle_skips_short_cycles(self):
        """
        If min_samples_per_cycle is very large, all cycles should be skipped
        (no fitted model), but every epoch cycle still gets a fit_ok=False
        row with NaN geometry rather than being dropped from the table.
        """
        X, phase_true, _ = _make_cyclic_3d(n_cycles=3, samples_per_cycle=50)
        epochs = _epochs(phase_true, 100.0)
        samples, cycles, details = fit_pca_phase_coordinates(
            X, epochs=epochs, min_samples_per_cycle=1000
        )
        assert len(details["models"]) == 0
        assert len(cycles) == epochs.n_cycles
        assert not cycles["fit_ok"].any()
        assert cycles["center_x"].isna().all()


# ---------------------------------------------------------------------------
# TestPublicContract
# ---------------------------------------------------------------------------

class TestPublicContract:
    def _make_tilted_circle(self, n=300, noise=0.02):
        t = np.linspace(0, 6 * np.pi, n)
        tilt = np.pi / 6
        X = np.column_stack([
            np.cos(t),
            np.sin(t) * np.cos(tilt),
            np.sin(t) * np.sin(tilt),
        ]) + np.random.default_rng(0).normal(0, noise, (n, 3))
        phase = t.copy()
        return X, phase

    def test_pca_returns_tuple3(self):
        X, phase = self._make_tilted_circle()
        result = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert len(result) == 3

    def test_pca_sample_columns(self):
        X, phase = self._make_tilted_circle()
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert list(samples.columns) == SAMPLE_COLUMNS

    def test_pca_cycle_columns(self):
        X, phase = self._make_tilted_circle()
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert list(cycles.columns) == CYCLE_COLUMNS

    def test_pca_details_algorithm(self):
        X, phase = self._make_tilted_circle()
        _, _, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert details["algorithm"] == "pca"
        assert "models" in details

    def test_reconstruct_pca_near_exact(self):
        X, phase = self._make_tilted_circle(noise=0.0)
        samples, cycles, details = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        X_hat = reconstruct_phase_coordinates(samples, cycles)
        fitted_mask = ~np.isnan(X_hat[:, 0])
        assert fitted_mask.sum() > 0
        np.testing.assert_allclose(X_hat[fitted_mask], X[fitted_mask], atol=1e-10)

    def test_sample_columns_identical_schema(self):
        """Both algorithms produce samples with the same column names."""
        from phase_coordinates import SAMPLE_COLUMNS as SC
        X, phase = self._make_tilted_circle()
        samples, _, _ = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert list(samples.columns) == SC

    def test_cycle_columns_identical_schema(self):
        from phase_coordinates import CYCLE_COLUMNS as CC
        X, phase = self._make_tilted_circle()
        _, cycles, _ = fit_pca_phase_coordinates(X, epochs=_epochs(phase, 100.0))
        assert list(cycles.columns) == CC

    def test_bayesian_import_without_pymc(self):
        """Package imports cleanly even without PyMC."""
        import phase_coordinates
        assert hasattr(phase_coordinates, 'fit_pca_phase_coordinates')

    def test_bayesian_clear_error_without_deps(self):
        """fit_bayesian_phase_coordinates raises ImportError if PyMC missing."""
        try:
            import pymc  # noqa
            pytest.skip("PyMC is installed; skip this test")
        except ImportError:
            pass
        from phase_coordinates import fit_bayesian_phase_coordinates
        X, _ = self._make_tilted_circle()
        with pytest.raises(ImportError):
            fit_bayesian_phase_coordinates(X, sampling_rate_hz=100.0)

    @pytest.mark.slow
    def test_bayesian_smoke(self):
        """Basic smoke test for Bayesian fit."""
        pytest.importorskip("pymc")
        from phase_coordinates import fit_bayesian_phase_coordinates, reconstruct_phase_coordinates
        rng = np.random.default_rng(0)
        fs = 100.0
        n_per_cycle = 100
        n_cycles = 4
        n = n_per_cycle * n_cycles
        t = np.arange(n) / fs
        tilt = np.pi / 6
        X = np.column_stack([
            np.cos(2 * np.pi * t),
            np.sin(2 * np.pi * t) * np.cos(tilt),
            np.sin(2 * np.pi * t) * np.sin(tilt),
        ])
        X += rng.normal(0, 0.02, X.shape)
        samples, cycles, details = fit_bayesian_phase_coordinates(
            X, sampling_rate_hz=fs, draws=100, tune=100, chains=2, random_seed=0
        )
        assert list(samples.columns) == SAMPLE_COLUMNS
        assert list(cycles.columns) == CYCLE_COLUMNS
        assert details["algorithm"] == "bayesian"
        assert "diagnostics" in details
        X_hat = reconstruct_phase_coordinates(samples, cycles)
        assert X_hat.shape == X.shape

    @pytest.mark.slow
    def test_bayesian_output_matches_shared_contract(self):
        """The Bayesian samples/cycles output must actually satisfy the
        same contract the PCA path does: preserve the input pandas index,
        use integer cycle assignment with -1 outside the fitted window (not
        float NaN), and never let adjacent cycles' sample ranges overlap."""
        pytest.importorskip("pymc")
        from phase_coordinates import fit_bayesian_phase_coordinates
        rng = np.random.default_rng(1)
        fs = 100.0
        n_per_cycle = 100
        n_cycles = 4
        n = n_per_cycle * n_cycles
        t = np.arange(n) / fs
        tilt = np.pi / 6
        X_arr = np.column_stack([
            np.cos(2 * np.pi * t),
            np.sin(2 * np.pi * t) * np.cos(tilt),
            np.sin(2 * np.pi * t) * np.sin(tilt),
        ])
        X_arr += rng.normal(0, 0.02, X_arr.shape)

        # Non-default index: neither the default RangeIndex(0, n) nor
        # sorted-from-zero, so silently falling back to a default index
        # would be detectable.
        custom_index = pd.RangeIndex(start=5000, stop=5000 + n, step=1)
        X = pd.DataFrame(X_arr, columns=["x", "y", "z"], index=custom_index)

        samples, cycles, details = fit_bayesian_phase_coordinates(
            X, sampling_rate_hz=fs, draws=100, tune=100, chains=2, random_seed=0
        )

        # 1. Index preserved, same as the PCA path.
        pd.testing.assert_index_equal(samples.index, X.index)

        # 2. Integer cycle assignment, -1 outside the fitted window --
        # not float NaN.
        assert samples["cycle"].dtype.kind == "i"
        assert (samples["cycle"] == -1).any(), (
            "expected at least one sample outside the fitted window for a "
            "4-cycle recording with real endpoint boundary uncertainty"
        )
        assert set(samples["cycle"].unique()) <= set(range(-1, len(cycles)))

        # 3. Adjacent cycles' sample ranges never overlap (half-open
        # [sample_start, sample_stop) with sample_stop of cycle k <=
        # sample_start of cycle k+1).
        starts = cycles["sample_start"].to_numpy()
        stops = cycles["sample_stop"].to_numpy()
        assert np.all(stops[:-1] <= starts[1:]), (
            f"adjacent cycles overlap: sample_stop={stops[:-1]} vs "
            f"next sample_start={starts[1:]}"
        )
        # sample_stop is one past the last sample *actually* assigned to
        # that cycle in the samples table, not merely floor(t_stop*fs)+1.
        for k in range(len(cycles)):
            member_idx = np.flatnonzero(samples["cycle"].to_numpy() == k)
            if member_idx.size:
                assert stops[k] == member_idx[-1] + 1
                assert starts[k] == member_idx[0]
