#!/usr/bin/env python3
"""Electrical vs mechanical time constants of the StampFly motor+prop.
StampFly モータ+プロペラの電気的時定数と機械的時定数の比較（先生の同定値を使用）。

Parameter provenance / 出所（すべて sf_motor/data/current_parameters.md, 2026-04-19）:
  R  = 0.593 Ω          (LCRメータ、Foster直列抵抗 = DC巻線抵抗)
  L  = 3.37 µH          (低周波の合計: L_inf 0.79 + L1 1.97 + L2 0.61 µH)
  Ke = Kt = 5.35e-4     (V·s/rad, V-I-ω 実測の合同最小二乗)
  B  = 0                (物理制約 B>=0 の最小二乗解)
  C_Q = 9.72e-11        (トルク直接計測、旧プロペラ)
  J_rotor = 3.30e-8     (モータ分解・コイル形状/質量からの推算)
  J_prop  = 2.01e-8     (ブレード切り分け・長方形近似の推算)
  J_total = 5.31e-8 kg·m²

NOTE / 要確認の食い違い:
  firmware (vehicle_old motor_model.cpp) は Jmp = 2.01e-8 を
  「rotor+prop COMBINED」とコメントしているが、sf_motor では 2.01e-8 は
  J_prop 単体で、合計は 5.31e-8。どちらが正か先生に確認する。
  → 本スクリプトは両方の J で計算して感度を示す。

Model / モデル（ホバ点まわりの線形化, B=0）:
  電気: L·di/dt = V·duty − R·i − Ke·ω      → τ_e = L/R
  機械: J·dω/dt = Kt·i − C_Q·ω²
  電気が速いので i ≈ (V·duty − Ke·ω)/R を代入:
    J·Δω̇ = (Kt/R)·ΔV − (Kt·Ke/R + 2·C_Q·ω0)·Δω
    τ_m = J / (Kt·Ke/R + 2·C_Q·ω0)
  減衰は「逆起電力による電気ブレーキ Kt·Ke/R」と「プロペラ空力 2·C_Q·ω0」の和。

Yaw zero (第2章/第4章) / ヨー零点との関係:
  τ_z = J / (2·C_Q·ω0)   ← 空力減衰のみで決まる
  τ_z / τ_m = 1 + Kt·Ke/(R·2·C_Q·ω0)   ← J に依存しない！
"""

import math

R = 0.593
L = (0.78813 + 1.96696 + 0.61103) * 1e-6
KE = KT = 5.35e-4
CQ = 9.72e-11
CT = 1.0e-8
M, G = 0.0368, 9.81
OMEGA0 = math.sqrt(M * G / 4 / CT)

tau_e = L / R
d_emf = KT * KE / R          # back-EMF damping [N·m·s/rad]
d_aero = 2 * CQ * OMEGA0     # propeller aero damping [N·m·s/rad]

print(f"ホバ回転数 ω0            : {OMEGA0:.0f} rad/s")
print(f"電気的時定数 τ_e = L/R   : {tau_e*1e6:.1f} µs")
print(f"減衰の内訳: 逆起電力 KtKe/R = {d_emf:.3e},  空力 2·CQ·ω0 = {d_aero:.3e}  [N·m·s/rad]")
print(f"           （逆起電力 : 空力 = {d_emf/(d_emf+d_aero)*100:.0f}% : {d_aero/(d_emf+d_aero)*100:.0f}%）")
print()
for label, J in (("J = 2.01e-8 (firmware Jmp)", 2.01e-8),
                 ("J = 5.31e-8 (sf_motor J_total)", 5.31e-8)):
    tau_m = J / (d_emf + d_aero)
    tau_z = J / d_aero
    print(f"{label}")
    print(f"  機械的時定数 τ_m : {tau_m*1e3:6.1f} ms   (τ_m/τ_e = {tau_m/tau_e:,.0f} 倍)")
    print(f"  ヨー零点     τ_z : {tau_z*1e3:6.1f} ms")
    print()
print(f"τ_z/τ_m = 1 + KtKe/(R·2CQω0) = {1 + d_emf/d_aero:.2f}  ← J に依存しない（頑健）")
