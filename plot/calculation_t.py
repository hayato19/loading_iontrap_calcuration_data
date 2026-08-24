import numpy as np

# ---- physical constants ----
kB   = 1.380649e-23        # J/K
hbar = 1.054571817e-34     # J·s

def T_ratio_with_and_without_COM(
    v: np.ndarray,
    m: float,
    Gamma,
    s0,
    N,
    j=2/5,
):
    Gamma = float(Gamma)
    s0 = float(s0)
    j = float(j)
    m = float(m)

    M = 5

    trap_f = 0.5e6
    w = 200 #サイクル幅(無次元)
    n_cycle = N // w  # サイクル個数(=記録数/サイクル幅)
    
    v2 = np.zeros_like(v)
    # v2 = v2[:n_sum]
    T = np.zeros_like(v)
    # T = T[:n_sum]


    for cycle_num in range(n_cycle):    # 計算サイクル番号(同一サイクルでTは一定とする)
        cycle_start = w * cycle_num     # cycle_num番目のサイクルの開始番号
        for l in range(w):     # サイクル内の数え上げ
            v2[cycle_start][:] += v[cycle_start + l][:] ** 2 # サイクル初めのv2にサイクル内全データのv^2を加算
                    
        T_cycle = m / kB * v2[cycle_start][:] / w # サイクルの平均運動温度
                
        for l in range(w):     # サイクル内の数え上げ
            T[cycle_start + l] = T_cycle
        

    # T = np.zeros_like(v)
    # T[:][:] = m / kB * v[:][:] ** 2

    T_min = (hbar * Gamma * np.sqrt(1.0 + s0) / (4.0 * kB)) * (1.0 + j)
    print(n_cycle)

    return T, T_min, w*(n_cycle-1)