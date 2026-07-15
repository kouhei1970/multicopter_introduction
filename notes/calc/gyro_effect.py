#!/usr/bin/env python3
"""Rotor gyroscopic effect on StampFly: with vs without, quantified.
ロータのジャイロ効果を「入れた場合/入れない場合」で数値比較する（先生の宿題）。

Physics / 物理:
  Total angular momentum L = I·ω_body + h_rotor. In the body frame:
    I·ω̇ + ω×(I·ω) + ω×h = τ
  The rotor term is ω×h, where h = J_mp·Σ(dir_i·ω_i)·ê_z.
  At hover with equal speeds, CW and CCW cancel: Σdir_i·ω_i = 0 → h = 0.
  h becomes nonzero when a yaw command splits CW/CCW speeds.
  ホバで等速なら CW/CCW の角運動量は相殺（h=0）。ヨー指令で速度が割れると h≠0 になり、
  機体が回転すると ω×h のトルクがロール/ピッチ軸に漏れ出す。

Sign conventions (NED/FRD): CW prop (viewed from above) spins about +z (down),
CCW about −z. Positive yaw torque needs CCW pair faster → net h_z < 0.

Two deliverables / 出力は2つ:
  1. Torque magnitude table across scenarios (yaw split × body rate)
     シナリオ別のジャイロトルクの大きさ（制御トルク上限との比較つき）
  2. A 0.4 s maneuver simulation (hard yaw split + roll pulse for 0.2 s),
     integrating full Euler equations + quaternion attitude, run twice
     (with / without ω×h), reporting the attitude difference.
     フルのオイラー方程式+クォータニオン姿勢で同一機動を2回積分し差を報告。

Neglected / 無視した項: ḣ (rotor acceleration reaction; acts on yaw only,
identical in both runs -> cancels in the comparison), aero damping,
translation coupling, thrust change from the roll pulse (superimposed torque).

Parameters: same provenance as roll_divergence.py, plus
  J_mp = 1.375e-8 kg·m² (2026-07-15確定値: プロペラ1.030e-8[写真+画素積分+ピッチ補正]
         + 回転子3.45e-9[実測諸元コップモデル]。旧5.31e-8は旧プロペラ+桁違いの過大)
  I = diag(9.16e-6, 13.3e-6, 20.4e-6) kg·m² (kSpecInertia)
"""

import math

# --- parameters ---
IXX, IYY, IZZ = 9.16e-6, 13.3e-6, 20.4e-6   # body inertia [kg m^2]
JMP = 1.375e-8       # rotor+prop total inertia [kg m^2] (2026-07-15確定: 写真法プロペラ1.030e-8+諸元法回転子3.45e-9)
CT = 6.7e-9          # thrust coefficient [N/(rad/s)^2] (現行プロペラ: ベンチ推力×実測ω)
KAPPA = 6.12e-3      # torque/thrust ratio [m] (現行プロペラ: C_Q/C_T = 4.10e-11/6.7e-9)
M, G, D = 0.0368, 9.81, 0.023
T0 = M * G / 4.0
OMEGA0 = math.sqrt(T0 / CT)          # hover prop speed [rad/s]
TAU_MAX = 4 * 0.25 * T0 * D          # control torque authority (±25% thrust split)

# ---------- 1. magnitude table ----------
print("== 1. ジャイロトルクの大きさ（h = 4·J_mp·Δω, τ_gyro = h × 機体レート）==")
print(f"   ホバ回転数 ω0 = {OMEGA0:.0f} rad/s, 制御トルクの目安 = {TAU_MAX*1e3:.2f} mN·m\n")
print(f"{'ヨー割合':>10} {'Δω [rad/s]':>12} {'h [N·m·s]':>12} | " +
      " | ".join(f"レート{p:>2} rad/s" for p in (1, 5, 10)))
for frac in (0.025, 0.125):
    dw = frac * OMEGA0
    h = 4 * JMP * dw
    cells = []
    for p in (1, 5, 10):
        tau = h * p
        cells.append(f"{tau*1e6:7.1f} µN·m ({100*tau/TAU_MAX:4.1f}%)")
    print(f"{frac*100:9.1f}% {dw:12.0f} {h:12.2e} | " + " | ".join(cells))
print("   ※ %は制御トルク上限に対する比率\n")

# ---------- 2. maneuver simulation ----------
# scenario: for t in [0, 0.2]s apply roll torque pulse 3e-4 N·m AND hold a
# hard yaw split (Δω = 12.5% of ω0). Then coast to 0.4 s.
# シナリオ: 最初の0.2秒間、ロールトルク3e-4 N·mと強いヨー指令（Δω=12.5%）を同時印加。
T_PULSE, T_END, DT = 0.2, 0.4, 1e-5
TAU_X_PULSE = 3e-4                       # [N m]
DW_PULSE = 0.125 * OMEGA0                # [rad/s]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def euler_zyx(q):
    """quaternion -> roll, pitch, yaw [rad] (ZYX)"""
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def simulate(with_gyro: bool):
    p = q_ = r = 0.0
    quat = (1.0, 0.0, 0.0, 0.0)
    t = 0.0
    snaps = {}
    while t < T_END - 1e-12:
        pulse = t < T_PULSE
        tau_x = TAU_X_PULSE if pulse else 0.0
        dw = DW_PULSE if pulse else 0.0
        # yaw torque from the split (air reaction, linear-exact form)
        # ヨートルク（空気反作用）: κ·2·Ct·[(ω0+Δω)²−(ω0−Δω)²] = 8κCtω0Δω
        tau_z = 8 * KAPPA * CT * OMEGA0 * dw
        h = -4 * JMP * dw          # CCW faster -> h_z negative / CCWが速いとhは負
        gx = q_ * h if with_gyro else 0.0    # (ω×h)_x = q·h
        gy = -p * h if with_gyro else 0.0    # (ω×h)_y = −p·h
        # Euler equations (diagonal I): I·ω̇ = τ − ω×(Iω) − ω×h
        dp = (tau_x - (IZZ - IYY) * q_ * r - gx) / IXX
        dq = (0.0 - (IXX - IZZ) * p * r - gy) / IYY
        dr = (tau_z - (IYY - IXX) * p * q_) / IZZ
        p += dp * DT
        q_ += dq * DT
        r += dr * DT
        # quaternion kinematics: q̇ = ½ q ⊗ (0, ω)
        dquat = quat_mul(quat, (0.0, p, q_, r))
        quat = tuple(quat[i] + 0.5 * dquat[i] * DT for i in range(4))
        n = math.sqrt(sum(c*c for c in quat))
        quat = tuple(c / n for c in quat)
        t += DT
        for ts in (T_PULSE, T_END):
            if abs(t - ts) < DT / 2:
                snaps[ts] = (p, q_, r, euler_zyx(quat))
    return snaps


runs = {label: simulate(g) for label, g in (("with", True), ("without", False))}

print("== 2. 機動シミュレーション（0.2秒間: ロールパルス3e-4 N·m + ヨー割り12.5%）==")
for ts in (T_PULSE, T_END):
    print(f"\n  t = {ts:.1f} s:")
    for label in ("with", "without"):
        p, q_, r, (ph, th, ps) = runs[label][ts]
        name = "ジャイロ項あり" if label == "with" else "ジャイロ項なし"
        print(f"    {name}: p={p:6.2f} q={q_:6.2f} r={r:6.2f} rad/s | "
              f"roll={math.degrees(ph):7.2f}° pitch={math.degrees(th):7.2f}° yaw={math.degrees(ps):7.2f}°")
    pw = runs["with"][ts]
    po = runs["without"][ts]
    dth = math.degrees(pw[3][1] - po[3][1])
    dph = math.degrees(pw[3][0] - po[3][0])
    print(f"    差（あり−なし）: Δroll={dph:+.2f}°  Δpitch={dth:+.2f}°")
