import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print("matplotlib is required: pip install matplotlib")
    raise

from compare_dat import resample_polyline, symmetric_nn_stats
from data_loader import load_airfoil_data


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(
            "Usage: python utils/plot_dat_compare.py <ref_dat> <test_dat> [num_samples=2000] [chord_mm]"
        )
        sys.exit(1)

    ref_path = sys.argv[1]
    test_path = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 2000
    chord_mm = float(sys.argv[4]) if len(sys.argv) == 5 else None

    u_ref, l_ref, name_ref, _ = load_airfoil_data(ref_path, logger_func=lambda *_: None)
    u_tst, l_tst, name_tst, _ = load_airfoil_data(test_path, logger_func=lambda *_: None)

    # Resample along arc length for fair comparison/plotting
    u_ref_r = resample_polyline(u_ref, num_samples)
    u_tst_r = resample_polyline(u_tst, num_samples)
    l_ref_r = resample_polyline(l_ref, num_samples)
    l_tst_r = resample_polyline(l_tst, num_samples)

    # Compute symmetric NN metrics (for titles)
    hu, ru, pu = symmetric_nn_stats(u_ref_r, u_tst_r)
    hl, rl, pl = symmetric_nn_stats(l_ref_r, l_tst_r)

    # One-way NN (ref→test) for plotting error vs x
    from scipy.spatial import cKDTree

    ut_tree = cKDTree(u_tst_r)
    lt_tree = cKDTree(l_tst_r)
    du = ut_tree.query(u_ref_r, k=1)[0]
    dl = lt_tree.query(l_ref_r, k=1)[0]

    # Optional conversion to mm
    if chord_mm is not None and chord_mm > 0:
        du_mm = du * chord_mm
        dl_mm = dl * chord_mm
    else:
        du_mm = dl_mm = None

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(u_ref_r[:, 0], u_ref_r[:, 1], 'k-', lw=1.5, label=f"Ref Upper ({name_ref})")
    ax0.plot(l_ref_r[:, 0], l_ref_r[:, 1], 'k-', lw=1.5, label=f"Ref Lower ({name_ref})")
    ax0.plot(u_tst_r[:, 0], u_tst_r[:, 1], color='tab:orange', lw=1.5, label=f"Test Upper ({name_tst})")
    ax0.plot(l_tst_r[:, 0], l_tst_r[:, 1], color='tab:orange', lw=1.5, label=f"Test Lower ({name_tst})")
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title('Overlay (resampled)')
    ax0.set_xlabel('x/c')
    ax0.set_ylabel('y/c')
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='best')

    ax1 = axes[1]
    ax1.plot(u_ref_r[:, 0], du, color='tab:blue', lw=1.5)
    ttl_u = f"Upper error (ref→test NN). RMS={ru:.2e}, H={hu:.2e} [x/c]"
    if du_mm is not None:
        ttl_u += f" | RMS={ru*chord_mm:.3f} mm, H={hu*chord_mm:.3f} mm"
    ax1.set_title(ttl_u)
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('distance [x/c]')
    ax1.grid(True, alpha=0.3)

    if du_mm is not None:
        ax1b = ax1.twinx()
        ax1b.set_ylabel('distance [mm]')
        ax1b.plot(u_ref_r[:, 0], du_mm, color='tab:gray', alpha=0.3)

    ax2 = axes[2]
    ax2.plot(l_ref_r[:, 0], dl, color='tab:green', lw=1.5)
    ttl_l = f"Lower error (ref→test NN). RMS={rl:.2e}, H={hl:.2e} [x/c]"
    if dl_mm is not None:
        ttl_l += f" | RMS={rl*chord_mm:.3f} mm, H={hl*chord_mm:.3f} mm"
    ax2.set_title(ttl_l)
    ax2.set_xlabel('x/c')
    ax2.set_ylabel('distance [x/c]')
    ax2.grid(True, alpha=0.3)

    if dl_mm is not None:
        ax2b = ax2.twinx()
        ax2b.set_ylabel('distance [mm]')
        ax2b.plot(l_ref_r[:, 0], dl_mm, color='tab:gray', alpha=0.3)

    plt.show()


if __name__ == '__main__':
    main()

