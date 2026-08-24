import argparse
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from plot.plot_full import plot_full_x
from plot.plot_t import plot_t
from plot.calculation_t import T_ratio_with_and_without_COM


plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 15


def load_metadata(run_dir):
    """
    runフォルダ内の metadata.json を読み込む。
    """
    metadata_path = os.path.join(run_dir, "metadata.json")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.json が見つかりません: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def get_data_path(run_dir, metadata, key):
    """
    metadata["files"] に記録されたファイル名から実ファイルのパスを得る。
    """
    try:
        file_name = metadata["files"][key]
    except KeyError as e:
        raise KeyError(
            f"metadata.json の files に '{key}' がありません"
        ) from e

    path = os.path.join(run_dir, file_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{key} データが見つかりません: {path}"
        )

    return path


def normalize_peak(y):
    """
    添付 load_rho_int.py の normalize_peak() と同じ考え方で、
    最大値が1になるように規格化する。
    """
    y = np.asarray(y, dtype=float)
    y_max = np.nanmax(y)

    if not np.isfinite(y_max) or y_max == 0:
        raise ValueError(
            "Cannot normalize: maximum value is zero or non-finite"
        )

    return y / y_max


def build_time_axis(n_points, dt_rec):
    """
    保存データの時間間隔 dt_rec から時間軸を再構成する。
    """
    return np.arange(n_points, dtype=float) * dt_rec


def plot_rho_int_delta(
        rho_int,
        delta,
        mode="each",
        save_dir="./figs",
):
    """
    添付 load_rho_int.py の描画ロジックを利用しつつ、
    横軸は保存済み delta.npy を直接使用する。

    delta は [rad/s] として保存されているため、
    delta / (2*pi) を Hz に変換し、さらに MHz にする。
    """
    rho_int = np.asarray(rho_int)
    delta = np.asarray(delta)

    if rho_int.ndim != 2:
        raise ValueError(
            f"rho_int must be 2D, got shape={rho_int.shape}"
        )

    if delta.ndim != 1:
        raise ValueError(
            f"delta must be 1D, got shape={delta.shape}"
        )

    if rho_int.shape[0] != len(delta):
        raise ValueError(
            "rho_int と delta の点数が一致しません: "
            f"rho_int={rho_int.shape[0]}, delta={len(delta)}"
        )

    _, M = rho_int.shape

    detuning_MHz = delta / (2.0 * math.pi) / 1e6

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
            linewidth=0.5,
        )

    elif mode == "mean":
        rho_plot = np.nanmean(rho_int, axis=1)
        rho_plot = normalize_peak(rho_plot)

        plt.plot(
            detuning_MHz,
            rho_plot,
            ".-",
            markersize=1.0,
            linewidth=0.5,
        )

    elif mode == "each":
        for k in range(M):
            rho_each = normalize_peak(rho_int[:, k])

            plt.plot(
                detuning_MHz,
                rho_each,
                ".-",
                markersize=1.0,
                linewidth=0.5,
                label=f"particle {k}",
            )

        if M > 1:
            plt.legend()

    else:
        raise ValueError(
            "mode must be 'sum', 'mean', or 'each'"
        )

    plt.xlabel("共鳴からの離調周波数(MHz)", fontsize=16)
    plt.ylabel("励起確率(a.u.)", fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("rho_int shape:", rho_int.shape)
    print("delta shape:", delta.shape)
    print("Saved rho_int-delta figure:", save_path)

    return save_path


def load_and_plot(run_dir, rho_mode="each"):
    """
    1つのrunフォルダを読み込み、
      1. x-t
      2. T-t
      3. rho_int-delta
    の3種類を描画する。
    """
    run_dir = os.path.abspath(run_dir)
    metadata = load_metadata(run_dir)

    # ======================================
    # metadata から必要な条件を取得
    # ======================================

    try:
        simulation = metadata["simulation"]
        motion = metadata["motion_parameters"]

        M = int(simulation["M"])
        dt_rec = float(simulation["dt_rec_s"])
        N = int(simulation["N"])

        m_arr = np.asarray(
            motion["m_arr_kg"],
            dtype=float,
        )
        gamma_arr = np.asarray(
            motion["gamma_arr_rad_s"],
            dtype=float,
        )
        S0_arr = np.asarray(
            motion["S0_arr"],
            dtype=float,
        )

    except KeyError as e:
        raise KeyError(
            f"metadata.json に必要な項目がありません: {e}"
        ) from e

    if len(m_arr) != M:
        raise ValueError(
            f"M と m_arr の長さが一致しません: "
            f"M={M}, len(m_arr)={len(m_arr)}"
        )

    if len(gamma_arr) != M:
        raise ValueError(
            f"M と gamma_arr の長さが一致しません: "
            f"M={M}, len(gamma_arr)={len(gamma_arr)}"
        )

    if len(S0_arr) != M:
        raise ValueError(
            f"M と S0_arr の長さが一致しません: "
            f"M={M}, len(S0_arr)={len(S0_arr)}"
        )

    # ======================================
    # dataファイル読み込み
    # ======================================

    x_path = get_data_path(run_dir, metadata, "x")
    v_path = get_data_path(run_dir, metadata, "v")
    rho_int_path = get_data_path(run_dir, metadata, "rho_int")
    delta_path = get_data_path(run_dir, metadata, "delta")

    # x, v は必要に応じてOS側でページング可能なよう mmap_mode="r"
    xM = np.load(x_path, mmap_mode="r")
    vM = np.load(v_path, mmap_mode="r")

    # rho_int, delta は比較的小さいため通常読み込み
    rho_int = np.load(rho_int_path)
    delta = np.load(delta_path)

    if xM.ndim != 2 or xM.shape[1] != M:
        raise ValueError(
            f"x のshapeが不正です: x.shape={xM.shape}, M={M}"
        )

    if vM.ndim != 2 or vM.shape[1] != M:
        raise ValueError(
            f"v のshapeが不正です: v.shape={vM.shape}, M={M}"
        )

    if xM.shape[0] != vM.shape[0]:
        raise ValueError(
            "x と v の時間点数が一致しません: "
            f"x={xM.shape[0]}, v={vM.shape[0]}"
        )

    # 保存済みデータの時間間隔 dt_rec から時間軸を再生成
    t = build_time_axis(
        n_points=xM.shape[0],
        dt_rec=dt_rec,
    )

    # 各runの図を同じ場所にまとめる
    fig_dir = os.path.join(run_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    print("======================================")
    print("Loading simulation result")
    print("======================================")
    print("run_dir :", run_dir)
    print("M       :", M)
    print("dt_rec  :", dt_rec, "s")
    print("x shape :", xM.shape)
    print("v shape :", vM.shape)
    print("rho_int :", rho_int.shape)
    print("delta   :", delta.shape)

    # ======================================
    # 1. x-t グラフ
    # 添付 plot_full.py の plot_full_x() をそのまま利用
    # ======================================

    plot_full_x(
        t=t,
        xM=xM,
        save_dir=fig_dir,
    )

    # ======================================
    # 2. T-t グラフ
    # main.py と同じ温度計算を行い、
    # 添付 plot_t.py の plot_t() をそのまま利用
    # ======================================

    T, T_min, n_sum = T_ratio_with_and_without_COM(
        v=vM,
        m=m_arr[0],
        Gamma=gamma_arr[0],
        s0=S0_arr[0],
        N=N,
    )

    plot_t(
        t=t,
        T=T,
        T_min=T_min,
        M=M,
        n_sum=n_sum,
        save_dir=fig_dir,
    )

    # ======================================
    # 3. rho_int-delta グラフ
    # 添付 load_rho_int.py の描画ロジックを踏襲
    # delta軸だけは保存済み delta.npy を直接使用
    # ======================================

    rho_plot_path = plot_rho_int_delta(
        rho_int=rho_int,
        delta=delta,
        mode=rho_mode,
        save_dir=fig_dir,
    )

    print("======================================")
    print("Finished")
    print("figures :", fig_dir)
    print("======================================")

    return {
        "metadata": metadata,
        "t": t,
        "x": xM,
        "v": vM,
        "rho_int": rho_int,
        "delta": delta,
        "rho_plot_path": rho_plot_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "metadata.json と保存済み data を読み込み、"
            "x-t, T-t, rho_int-delta グラフを生成する。"
        )
    )

    parser.add_argument(
        "run_dir",
        help=(
            "metadata.json, x.npy, v.npy, rho_int.npy, delta.npy "
            "を含むrunフォルダ"
        ),
    )

    parser.add_argument(
        "--rho-mode",
        choices=["sum", "mean", "each"],
        default="each",
        help="rho_int-delta グラフの描画方法（default: each）",
    )

    args = parser.parse_args()

    load_and_plot(
        run_dir=args.run_dir,
        rho_mode=args.rho_mode,
    )


if __name__ == "__main__":
    main()
