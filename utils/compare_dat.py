import sys
import numpy as np
from scipy.spatial import cKDTree

from data_loader import load_airfoil_data


def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to a fixed number of points using arc-length parameterization."""
    pts = np.asarray(points, float)
    if len(pts) <= 2 or num_points <= 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(pts[:1], num_points, axis=0)
    t = s / s[-1]
    tq = np.linspace(0.0, 1.0, num_points)
    xq = np.interp(tq, t, pts[:, 0])
    yq = np.interp(tq, t, pts[:, 1])
    return np.column_stack([xq, yq])


def symmetric_nn_stats(a: np.ndarray, b: np.ndarray):
    ta = cKDTree(a)
    tb = cKDTree(b)
    d1 = tb.query(a, k=1)[0]
    d2 = ta.query(b, k=1)[0]
    d = np.concatenate([d1, d2])
    hausdorff = float(np.max(d))
    rms = float(np.sqrt(np.mean(d ** 2)))
    p95 = float(np.percentile(d, 95.0))
    return hausdorff, rms, p95


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python utils/compare_dat.py <ref_dat> <test_dat> [num_samples_per_surface=2000]")
        sys.exit(1)

    ref_path, test_path = sys.argv[1], sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) == 4 else 2000

    u_ref, l_ref, _, _ = load_airfoil_data(ref_path, logger_func=lambda *_: None)
    u_tst, l_tst, _, _ = load_airfoil_data(test_path, logger_func=lambda *_: None)

    # Resample both pairs to uniform arc-length grids
    u_ref_r = resample_polyline(u_ref, num_samples)
    u_tst_r = resample_polyline(u_tst, num_samples)
    l_ref_r = resample_polyline(l_ref, num_samples)
    l_tst_r = resample_polyline(l_tst, num_samples)

    hu, ru, pu = symmetric_nn_stats(u_ref_r, u_tst_r)
    hl, rl, pl = symmetric_nn_stats(l_ref_r, l_tst_r)
    ha, ra, pa = symmetric_nn_stats(np.vstack([u_ref_r, l_ref_r]), np.vstack([u_tst_r, l_tst_r]))

    print("Comparison (normalized chord units, resampled):")
    print(f"Upper   → Hausdorff: {hu:.6e}, RMS: {ru:.6e}, P95: {pu:.6e}")
    print(f"Lower   → Hausdorff: {hl:.6e}, RMS: {rl:.6e}, P95: {pl:.6e}")
    print(f"Combined→ Hausdorff: {ha:.6e}, RMS: {ra:.6e}, P95: {pa:.6e}")


if __name__ == "__main__":
    main()

