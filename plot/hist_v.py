import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- physical constants ----
kB = 1.380649e-23          # Boltzmann constant [J/K]
NA = 6.02214076e23         # Avogadro constant [1/mol]

# 9Be+ ion mass [kg]
m = 9.0e-3 / NA


def gaussian_mb(v, A, T):
    """
    Gaussian velocity distribution assuming thermal equilibrium:

        f(v) = A * exp(-m v^2 / (2 kB T))
    """
    return A * np.exp(-m * v**2 / (2.0 * kB * T))


def Hist_v(
    v: np.ndarray,
    time: np.ndarray,
    bin_divisions: int = 20
):
    """
    Plot a velocity histogram and Gaussian fit for each particle,
    and estimate the temperature of each particle.

    Assumed input shape
    -------------------
    v.shape == (n_time, n_particles)

    For 3 particles:
        v[:, 0] : particle 1 velocity
        v[:, 1] : particle 2 velocity
        v[:, 2] : particle 3 velocity

    Parameters
    ----------
    v : np.ndarray
        Velocity array with shape (n_time, n_particles) [m/s].

    time : np.ndarray
        Retained for compatibility. Not used directly here.

    bin_divisions : int, default=20
        Number of equal-width bins between min(v_i) and max(v_i)
        for each particle i.

    Returns
    -------
    T_fit_all : np.ndarray
        Fitted temperatures [K].

    T_err_all : np.ndarray
        1-sigma fitting errors [K].
    """

    v = np.asarray(v, dtype=float)

    if v.ndim != 2:
        raise ValueError(
            f"v must be a 2D array with shape (n_time, n_particles). "
            f"Received shape: {v.shape}"
        )

    if bin_divisions < 2:
        raise ValueError("bin_divisions must be 2 or larger.")

    _, n_particles = v.shape

    T_fit_all = np.full(n_particles, np.nan, dtype=float)
    T_err_all = np.full(n_particles, np.nan, dtype=float)

    # ---- figure ----
    fig, axes = plt.subplots(
        n_particles,
        1,
        figsize=(8, 4 * n_particles),
        squeeze=False
    )

    axes = axes[:, 0]

    # ============================================================
    # particle-by-particle calculation
    # ============================================================
    for i in range(n_particles):

        ax = axes[i]

        # ---- velocity data of particle i ----
        v_data = v[:, i]
        v_data = v_data[np.isfinite(v_data)]

        if v_data.size < 3:
            ax.text(
                0.5,
                0.5,
                f"Particle {i + 1}\nNot enough valid data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Counts")

            continue

        # ========================================================
        # histogram
        # ========================================================

        v_min = np.min(v_data)
        v_max = np.max(v_data)

        if np.isclose(v_min, v_max):

            ax.text(
                0.5,
                0.5,
                f"Particle {i + 1}\nAll velocities are nearly identical",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Counts")

            continue

        # [v_min, v_max] を bin_divisions 等分
        bin_edges = np.linspace(
            v_min,
            v_max,
            bin_divisions + 1
        )

        counts, _ = np.histogram(
            v_data,
            bins=bin_edges
        )

        bin_centers = 0.5 * (
            bin_edges[:-1] + bin_edges[1:]
        )

        # ========================================================
        # fitting data
        # ========================================================

        # count = 0 のbinはフィッティングから除外
        mask = counts > 0

        x_fit = bin_centers[mask]
        y_fit = counts[mask]

        if x_fit.size < 2:

            ax.hist(
                v_data,
                bins=bin_edges,
                alpha=0.7,
                label="Velocity histogram"
            )

            ax.set_xlim(v_min, v_max)
            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Counts")

            ax.set_title(
                f"Particle {i + 1}: insufficient bins for fitting"
            )

            ax.legend()

            continue

        # ========================================================
        # initial values
        # ========================================================

        A0 = np.max(y_fit)

        # <v^2> = kB T / m を初期値として使用
        T0 = m * np.mean(v_data**2) / kB

        T0 = max(T0, 1.0e-12)

        # ========================================================
        # Gaussian fitting
        # ========================================================

        try:

            popt, pcov = curve_fit(
                gaussian_mb,
                x_fit,
                y_fit,
                p0=(A0, T0),
                bounds=(
                    [0.0, 1.0e-12],
                    [np.inf, np.inf]
                ),
                maxfev=20000
            )

            A_fit, T_fit = popt

            # ---- 1-sigma error ----
            if (
                pcov.shape == (2, 2)
                and np.isfinite(pcov[1, 1])
                and pcov[1, 1] >= 0
            ):

                T_err = np.sqrt(pcov[1, 1])

            else:

                T_err = np.nan

            # save results
            T_fit_all[i] = T_fit
            T_err_all[i] = T_err

            # ====================================================
            # fitting curve
            # ====================================================

            v_plot = np.linspace(
                v_min,
                v_max,
                1000
            )

            y_plot = gaussian_mb(
                v_plot,
                A_fit,
                T_fit
            )

            # ====================================================
            # plot
            # ====================================================

            ax.hist(
                v_data,
                bins=bin_edges,
                alpha=0.7,
                label="Velocity histogram"
            )

            ax.plot(
                v_plot,
                y_plot,
                linewidth=2,
                label="Gaussian fit"
            )

            ax.set_xlim(v_min, v_max)

            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Counts")

            # ---- temperature in title ----
            if np.isfinite(T_err):

                ax.set_title(
                    f"Particle {i + 1}: "
                    f"T = {T_fit * 1e3:.6g} "
                    f"± {T_err * 1e3:.2g} mK"
                )

            else:

                ax.set_title(
                    f"Particle {i + 1}: "
                    f"T = {T_fit * 1e3:.6g} mK"
                )

            ax.legend()

        # ========================================================
        # fitting failure
        # ========================================================
        except (RuntimeError, ValueError) as e:

            ax.hist(
                v_data,
                bins=bin_edges,
                alpha=0.7,
                label="Velocity histogram"
            )

            ax.set_xlim(v_min, v_max)

            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Counts")

            ax.set_title(
                f"Particle {i + 1}: fitting failed"
            )

            ax.legend()

            print(
                f"Particle {i + 1}: fitting failed: {e}"
            )

    # ============================================================
    # overall figure
    # ============================================================

    fig.suptitle(
        "Velocity Distribution and Gaussian Fit for Each Particle",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # ============================================================
    # print fitted temperatures
    # ============================================================

    print()
    print("===== Fitted temperatures =====")

    for i in range(n_particles):

        T_fit = T_fit_all[i]
        T_err = T_err_all[i]

        if np.isfinite(T_fit):

            if np.isfinite(T_err):

                print(
                    f"Particle {i + 1}: "
                    f"T = {T_fit:.6e} K "
                    f"({T_fit * 1e3:.6e} mK), "
                    f"1-sigma error = {T_err:.6e} K"
                )

            else:

                print(
                    f"Particle {i + 1}: "
                    f"T = {T_fit:.6e} K "
                    f"({T_fit * 1e3:.6e} mK)"
                )

        else:

            print(
                f"Particle {i + 1}: "
                f"temperature could not be determined"
            )

    return T_fit_all, T_err_all