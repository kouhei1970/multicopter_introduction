#!/usr/bin/env python3
"""Rotor (coil cup + shaft) inertia from measured specs.
コアレスモータ回転部（コイルかご+軸）の実測諸元からの慣性モーメント計算。

Specs (teacher, 2026-07-15) / 先生提供の実測諸元:
  コイルかご: 外径6mm、肉厚0.65mm、高さ12.4mm、上面が開き底面が閉じたコップ形状
  軸: 鉄、直径0.8mm、長さ21mm
  軸+コイルの総質量: 0.58 g

Model / モデル:
  かご = 側壁（中空円筒）+ 底（円盤、肉厚は側壁と同じ0.65mmと仮定）
  質量配分: 軸は鉄密度7870 kg/m³で質量を計算し、残りをかごに配分。
  かご内では体積比で側壁/底に配分（コイル線+含浸の実効密度が一様と仮定）。
  I(中空円筒) = m(Ro²+Ri²)/2, I(円盤) = mR²/2, I(軸) = mr²/2
"""

import math

M_TOTAL = 0.58e-3          # [kg]
RO = 3.0e-3                # cup outer radius
T_WALL = 0.65e-3
RI = RO - T_WALL
H_CUP = 12.4e-3
T_BOT = T_WALL             # bottom thickness assumption / 底の厚み=肉厚と仮定
R_SHAFT = 0.4e-3
L_SHAFT = 21e-3
RHO_FE = 7870.0

# shaft mass from iron density / 軸の質量（鉄密度から）
V_shaft = math.pi * R_SHAFT**2 * L_SHAFT
m_shaft = RHO_FE * V_shaft
m_cup = M_TOTAL - m_shaft
print(f"shaft: V={V_shaft*1e9:.2f} mm³, m={m_shaft*1e3:.4f} g")
print(f"cup (coil): m = {m_cup*1e3:.4f} g")

# cup volume split / かご体積の側壁/底への配分
h_wall = H_CUP - T_BOT
V_wall = math.pi * (RO**2 - RI**2) * h_wall
V_bot = math.pi * (RO**2 - R_SHAFT**2) * T_BOT
m_wall = m_cup * V_wall / (V_wall + V_bot)
m_bot = m_cup * V_bot / (V_wall + V_bot)
rho_eff = m_cup / (V_wall + V_bot)
print(f"wall: V={V_wall*1e9:.1f} mm³ m={m_wall*1e3:.4f} g / bottom: V={V_bot*1e9:.1f} mm³ m={m_bot*1e3:.4f} g")
print(f"effective coil density: {rho_eff:.0f} kg/m³ (銅8960の{100*rho_eff/8960:.0f}% — 巻線+空隙として妥当か確認)")

I_wall = 0.5 * m_wall * (RO**2 + RI**2)
I_bot = 0.5 * m_bot * (RO**2 + R_SHAFT**2)
I_shaft = 0.5 * m_shaft * R_SHAFT**2
J_ROTOR = I_wall + I_bot + I_shaft
print(f"\nI_wall  = {I_wall:.3e}")
print(f"I_bot   = {I_bot:.3e}")
print(f"I_shaft = {I_shaft:.3e}  (無視できる)")
print(f"★ J_rotor = {J_ROTOR:.3e} kg·m²  （旧値 3.30e-8 の {J_ROTOR/3.30e-8*100:.0f}% ≒ 1/10 → 桁違い説を確認）")

# ---- combine with confirmed prop inertia and coast-down ----
J_PROP = 1.030e-8          # pixel-integration + pitch correction (確定値)
J_OVER_CQ = 335.3          # coast-down (実測)
J_TOTAL = J_ROTOR + J_PROP
CQ = J_TOTAL / J_OVER_CQ
print(f"\nJ_total = J_rotor + J_prop = {J_TOTAL:.3e} kg·m²")
print(f"C_Q = J_total / (J/C_Q) = {CQ:.3e} N·m·s²/rad²")

# derived chain (needs Q4-10: bench thrust pairing) / 派生値（Q4-10前提）
CT = 6.7e-9                # bench T(duty0.1)=1.0gf × ω=1209rad/s pairing
M, G = 0.0368, 9.81
W_H = math.sqrt(M * G / 4 / CT)
KT = KE = 5.35e-4
R_W = 0.593
B_FULL = KT * KE / R_W
kappa = CQ / CT
tau_z = J_OVER_CQ / (2 * W_H)
tau_eff = J_OVER_CQ / (B_FULL / CQ + 2 * W_H)   # γ=1 (hover CCM)
print(f"\n-- 派生値（C_T=6.7e-9 のベンチペアリング前提、Q4-10確認待ち） --")
print(f"ω_hover = {W_H:.0f} rad/s ({W_H*60/2/math.pi:,.0f} rpm)")
print(f"κ = C_Q/C_T = {kappa*1e3:.2f} mm  (旧プロペラ 9.71mm)")
print(f"τ_z(ホバ)  = {tau_z*1e3:.1f} ms")
print(f"τ_eff(ホバ, γ=1) = {tau_eff*1e3:.1f} ms  ← firmware/lesson の 20ms と比較")
print(f"τ_z/τ_eff = {tau_z/tau_eff:.2f}")
