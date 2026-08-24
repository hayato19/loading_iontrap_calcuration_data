import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False


def calculate_fwhm(x, y):
    """
    単峰スペクトル y(x) の FWHM を計算する。
    x: detuning 軸 [MHz]
    y: 規格化スペクトル
    """

    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Not enough valid data points to calculate FWHM")

    peak_index = np.argmax(y)
    y_max = y[peak_index]
    half_max = y_max / 2.0

    left_indices = np.where(y[:peak_index] < half_max)[0]
    if len(left_indices) == 0:
        raise ValueError("Left half-maximum point not found")

    i1 = left_indices[-1]
    i2 = i1 + 1
    x_left = x[i1] + (half_max - y[i1]) * (x[i2] - x[i1]) / (y[i2] - y[i1])

    right_indices = np.where(y[peak_index:] < half_max)[0]
    if len(right_indices) == 0:
        raise ValueError("Right half-maximum point not found")

    i2 = peak_index + right_indices[0]
    i1 = i2 - 1
    x_right = x[i1] + (half_max - y[i1]) * (x[i2] - x[i1]) / (y[i2] - y[i1])

    fwhm = x_right - x_left

    return fwhm, x_left, x_right, half_max


def normalize_peak(y):
    """
    配列 y の最大値が 1 になるように規格化する。
    NaN は無視して最大値を計算する。
    """

    y = np.asarray(y, dtype=float)
    y_max = np.nanmax(y)

    if not np.isfinite(y_max) or y_max == 0:
        raise ValueError("Cannot normalize: maximum value is zero or non-finite")

    return y / y_max


def load_rho_int_and_plot(
        rho_int_path,
        detuning_num,
        mode="sum",
        save_dir="./figs"
):
    rho_int = np.load(rho_int_path)

    if rho_int.shape[0] != detuning_num:
        raise ValueError(
            f"detuning_num mismatch: "
            f"given={detuning_num}, file={rho_int.shape[0]}"
        )

    _, M = rho_int.shape

    scale = 0.5e7 * 2 * math.pi

    detuning_MHz = np.linspace(
        -scale / (2 * math.pi),
        scale / (2 * math.pi),
        detuning_num
    ) / 1e6

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"spectroscopy_from_rho_int_{mode}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(8, 5))

    if mode == "sum":
        rho_plot = np.nansum(rho_int, axis=1)
        rho_plot = normalize_peak(rho_plot)

        plt.plot(
            detuning_MHz,
            rho_plot,
            ".-",
            markersize=1.0,
            linewidth=0.5
        )

    elif mode == "mean":
        rho_plot = np.nanmean(rho_int, axis=1)
        rho_plot = normalize_peak(rho_plot)

        plt.plot(
            detuning_MHz,
            rho_plot,
            ".-",
            markersize=1.0,
            linewidth=0.5
        )

    elif mode == "each":
        # FWHM計算には全粒子和を使用
        rho_plot = np.nansum(rho_int, axis=1)
        rho_plot = normalize_peak(rho_plot)

        for k in range(M):
            rho_each = normalize_peak(rho_int[:, k])

            plt.plot(
                detuning_MHz,
                rho_each,
                ".-",
                markersize=1.0,
                linewidth=0.5
            )

    else:
        raise ValueError("mode must be 'sum', 'mean', or 'each'")

    # fwhm_MHz, x_left_MHz, x_right_MHz, half_max = calculate_fwhm(
    #     detuning_MHz,
    #     rho_plot
    # )

    plt.xlabel("共鳴からの離調周波数(MHz)", fontsize=16)
    plt.ylabel("励起確率(a.u.)",fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Loaded:", rho_int_path)
    print("rho_int shape:", rho_int.shape)
    # print("FWHM [MHz]:", fwhm_MHz)
    # print("FWHM [Hz]:", fwhm_MHz * 1e6)
    # print("left half max [MHz]:", x_left_MHz)
    # print("right half max [MHz]:", x_right_MHz)
    # print("half max:", half_max)
    print("Saved:", save_path)

    return rho_int, save_path
    # return rho_int, fwhm_MHz, save_path


if __name__ == "__main__":
    rho_int_path = "../data/rho_int_20260728_130500.npy"

    detuning_num = 3000

    load_rho_int_and_plot(
        rho_int_path=rho_int_path,
        detuning_num=detuning_num,
        mode="each"
    )