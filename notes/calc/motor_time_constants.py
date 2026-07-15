#!/usr/bin/env python3
"""Electrical vs mechanical time constants of the StampFly motor+prop.
StampFly モータ+プロペラの電気的時定数と機械的時定数の比較。

Authoritative source / 一次情報（先生指定, 2026-07-15）:
  sf_sandbox/paper/sf_motordriver.tex（モータドライバ論文）の同定値:
    R_s = 0.593 Ω        (DC巻線抵抗, LCRメータ/Fosterモデル同定)
    L_s = 0.788 µH       (直列インダクタンス)
    Ke = Kt = 5.35e-4    (V·s/rad, 無負荷V-ω実測の線形回帰, B≥0制約)
    C_Q = 9.72e-11       (トルク直接計測)
    J   = 5.31e-8 kg·m²  (Total inertia: モータ分解による回転子推算 3.30e-8
                          + ブレード切り分けによるプロペラ推算 2.01e-8)
    論文記載: τ_e = L_s/R_s = 1.33 µs, τ_m ≈ 110 ms

NOTE: firmware (vehicle_old motor_model.cpp) の Jmp=2.01e-8 は
「rotor+prop COMBINED」とコメントされているが、論文により 2.01e-8 は
プロペラ単体と判明（合計は 5.31e-8）。firmware の時定数 0.02 s との整合は
未解決 → 先生が計画中のフォトインタラプタ過渡測定が直接の決着手段。

Definitions / 時定数の定義の区別（混同しない）:
  τ_e  = L_s/R_s                          電気的時定数
  τ_m  = J·R_s/(Kt·Ke)                    古典的な機械的時定数（無負荷・起動スケール）
  τ_eff = J/(Kt·Ke/R_s + 2·C_Q·ω0)        ホバ点線形化の実効時定数（プロペラ負荷込み）
  τ_z  = J/(2·C_Q·ω0)                     ヨー軸の零点時定数（空力減衰のみ）
"""

import math

R = 0.593
LS = 0.788e-6
KE = KT = 5.35e-4
CQ = 9.72e-11
CT = 1.0e-8
J = 5.31e-8
M, G = 0.0368, 9.81
OMEGA0 = math.sqrt(M * G / 4 / CT)

tau_e = LS / R
tau_m_classic = J * R / (KT * KE)
d_emf = KT * KE / R
d_aero = 2 * CQ * OMEGA0
tau_eff = J / (d_emf + d_aero)
tau_z = J / d_aero

print(f"ホバ回転数 ω0                : {OMEGA0:.0f} rad/s")
print(f"電気的時定数 τ_e = L_s/R_s   : {tau_e*1e6:.2f} µs   （論文: 1.33 µs）")
print(f"機械的時定数 τ_m = J·R/(KtKe): {tau_m_classic*1e3:.0f} ms    （論文: ~110 ms）")
print(f"比 τ_m/τ_e                   : {tau_m_classic/tau_e:,.0f} 倍（約8万倍）")
print()
print(f"減衰の内訳（ホバ点）: 逆起電力 KtKe/R = {d_emf:.3e}, 空力 2·CQ·ω0 = {d_aero:.3e}")
print(f"                     （{d_emf/(d_emf+d_aero)*100:.0f}% : {d_aero/(d_emf+d_aero)*100:.0f}%）")
print(f"ホバ点の実効時定数 τ_eff     : {tau_eff*1e3:.1f} ms")
print(f"ヨー零点 τ_z = J/(2CQω0)     : {tau_z*1e3:.1f} ms")
print(f"τ_z/τ_eff = 1 + KtKe/(R·2CQω0) = {tau_z/tau_eff:.2f}（J に依存しない）")
