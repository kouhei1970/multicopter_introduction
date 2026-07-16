#!/usr/bin/env python3
"""What does each integrator in the cascade absorb? (Q5-5 follow-up)
カスケードの内側I・外側Iがそれぞれ何を吸収するかを分離して見せる（Q5-5の確認）。

先生の回答: 「内側にもIは入っています。外側にも場合によっては入れます」
実機 firmware/vehicle の実装: レート PID Ti=0.7s、姿勢 PID Ti=2.0s（両方入っている）。

仮説（学生の理解）:
  内側の I … 定常「トルク」外乱を吸収する（トリムずれ・重心オフセット・個体差トルク）。
             P-P だと角度に定常オフセットが残る: e = d/(K·Kp_r·Kp_a)
  外側の I … 「レート計測のバイアス」（ジャイロバイアス等）を吸収する。
             内側 I があっても、内側は『計測された』レートを指令に一致させるだけなので、
             真のレートには −bias が残り、角度に e = bias/Kp_a が残る。
これを別々の実験で確認する。

Model: cascade_bandwidth_study.py と同じ（K=102, τ=0.02s, L=12ms, ζ=0.7,
       Kp_r=0.245, 帯域比 1/3 → Kp_a ≈ 11.8）。Ti は実機値（内0.7s / 外2.0s）。
"""

import numpy as np

K = 102.0
TAU = 0.02
ZETA = 0.7
KP_R = 1.0 / (4 * ZETA**2 * K * TAU)
W_IN = np.sqrt(KP_R * K / TAU)
KP_A = W_IN / 3.0                     # 実証済みのスイートスポット / validated sweet spot
L = 0.012
DT = 2e-4


def simulate(t_end, ti_r=None, ti_a=None, d_acc=0.0, d_t0=1e9, bias=0.0, th_ref=0.0):
    """cascade sim with optional inner/outer I, torque disturbance, gyro bias.
    内側/外側I・トルク外乱・ジャイロバイアス付きカスケード応答"""
    n = int(t_end / DT)
    nd = int(round(L / DT))
    ubuf = np.zeros(nd + 1)
    m = r = th = 0.0
    int_r = int_a = 0.0
    out_th = np.empty(n)
    out_r = np.empty(n)
    for i in range(n):
        t = i * DT
        r_meas = r + bias                       # gyro sees true rate + bias
        e_a = th_ref - th
        if ti_a:
            int_a += e_a * DT
            r_cmd = KP_A * (e_a + int_a / ti_a)
        else:
            r_cmd = KP_A * e_a
        e_r = r_cmd - r_meas
        if ti_r:
            int_r += e_r * DT
            u = KP_R * (e_r + int_r / ti_r)
        else:
            u = KP_R * e_r
        ubuf[1:] = ubuf[:-1]
        ubuf[0] = u
        u_eff = ubuf[-1]
        m += (u_eff - m) / TAU * DT
        acc = K * m + (d_acc if t >= d_t0 else 0.0)   # disturbance = angular accel offset
        r += acc * DT
        th += r * DT
        out_th[i] = th
        out_r[i] = r
    return out_th, out_r


DEG = 180 / np.pi
TI_R, TI_A = 0.7, 2.0                 # firmware/vehicle 実機値 / real firmware values
print(f"Kp_r={KP_R:.3f}, Kp_a={KP_A:.1f} (比率1/3), L={L*1e3:.0f}ms, Ti内={TI_R}s Ti外={TI_A}s\n")

# -- 実験A: 定常トルク外乱（トリムずれ相当）d = 10 rad/s² を t=1.5s で印加 --
D = 10.0
thA_pp, _ = simulate(6.0, d_acc=D, d_t0=1.5)                     # P-P
thA_ip, _ = simulate(6.0, ti_r=TI_R, d_acc=D, d_t0=1.5)          # PI-P（内側Iのみ）
pred_A = D / (K * KP_R * KP_A) * DEG
print("実験A: 定常トルク外乱 10 rad/s²（トリムずれ相当）")
print(f"  P-P   最終角度 = {thA_pp[-1]*DEG:6.3f}°   (理論オフセット {pred_A:.3f}°)")
print(f"  PI-P  最終角度 = {thA_ip[-1]*DEG:6.3f}°   → 内側Iが吸収\n")

# -- 実験B: ジャイロバイアス 0.1 rad/s（≈5.7°/s、未校正相当）を最初から --
B = 0.1
thB_ip, _ = simulate(8.0, ti_r=TI_R, bias=B)                     # PI-P
thB_ii, _ = simulate(8.0, ti_r=TI_R, ti_a=TI_A, bias=B)          # PI-PI（外側Iも）
pred_B = -B / KP_A * DEG              # 機体は「見かけ0」= 真値−bias で釣り合う → 負側
print("実験B: ジャイロバイアス 0.1 rad/s（内側Iあり）")
print(f"  PI-P  最終角度 = {thB_ip[-1]*DEG:6.3f}°   (理論オフセット {pred_B:.3f}°)")
print(f"  PI-PI 最終角度 = {thB_ii[-1]*DEG:6.3f}°   → 外側Iが吸収\n")

# -- 位相コスト: I を入れる代償（交差周波数での位相遅れ）--
lag_in = np.degrees(np.arctan(1 / (W_IN * TI_R)))
lag_out = np.degrees(np.arctan(1 / (KP_A * TI_A)))
print(f"位相コスト: 内側I atan(1/(ω_in·Ti)) = {lag_in:.1f}°, 外側I = {lag_out:.1f}°")
print("→ 実機の Ti はどちらも交差周波数の1桁下: 位相余裕をほぼ食わずに定常誤差だけ消す設計\n")

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for cand in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP"):
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
tA = np.arange(len(thA_pp)) * DT
axes[0].plot(tA, thA_pp * DEG, "r-", label="P-P（Iなし）")
axes[0].plot(tA, thA_ip * DEG, "b-", label="PI-P（内側Iあり・実機Ti=0.7s）")
axes[0].axhline(pred_A, color="r", ls=":", lw=1)
axes[0].text(5.9, pred_A + 0.06, f"理論値 {pred_A:.1f}°", color="r", ha="right", fontsize=9)
axes[0].axvline(1.5, color="gray", ls="--", lw=0.8)
axes[0].text(1.55, -0.35, "外乱印加", color="gray", fontsize=9)
axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("角度 [deg]")
axes[0].set_title("実験A: 定常トルク外乱（トリムずれ）→ 内側Iの仕事")
axes[0].legend(); axes[0].grid(alpha=0.3)
tB = np.arange(len(thB_ip)) * DT
axes[1].plot(tB, thB_ip * DEG, "b-", label="PI-P（外側Iなし）")
axes[1].plot(tB, thB_ii * DEG, "g-", label="PI-PI（外側Iあり・実機Ti=2.0s）")
axes[1].axhline(pred_B, color="b", ls=":", lw=1)
axes[1].text(7.9, pred_B - 0.06, f"理論値 −b/Kp_a = {pred_B:.2f}°", color="b", ha="right", fontsize=9)
axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("角度 [deg]")
axes[1].set_title("実験B: ジャイロバイアス → 外側Iの仕事")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("notes/calc/cascade_integrator_study.png", dpi=130)
print("plot saved: notes/calc/cascade_integrator_study.png")
