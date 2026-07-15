#!/usr/bin/env python3
"""Coast-down identification of J/C_Q (and J) — the electrical-model-free test.
コーストダウン試験による J/C_Q（および J）の同定。電気駆動モデル不要の決着試験。

Data / データ: data/stampfly_prop_round_test.wfm（先生提供 2026-07-15）
  2 MSa/s, 6 s。約2.9sから Duty10% で約2秒駆動 → Duty0%（電源カット）→ 惰性減速。
  フォトインタラプタのディップは駆動定常で4枚全て検出可能
  （深さパターン [3.51, 3.46, 3.24(テープ), 3.56] V、間隔 ~1.295 ms）。

Physics / 物理:
  Duty0% ではローサイドFETが開き、モータ端子はCと繋がるのみ
  （Cは逆起電力電圧まで充電された後は i = C·dV/dt ≈ 0）→ 電気ブレーキなし。
  減速則: J·ω̇ = −C_Q·ω² − B·ω − τ_c
           （空力2乗 + 粘性摩擦 + クーロン摩擦）
  ω̇ = −a·ω² − b·ω − c とフィットすれば a = C_Q/J が電気モデル抜きで決まる。
  純空力なら 1/ω(t) が直線（傾き a）。摩擦があると低速側で直線から上に折れる。

Key outputs / 主要出力:
  - J/C_Q = 1/a（コーストダウンが直接測る量）
  - J = C_Q/a（C_Q = 9.72e-11 は論文のトルク直接計測値。旧プロペラ値である点に注意）
  - ヨー零点 τ_z(ホバ) = (J/C_Q)/(2·ω_hover) ← C_Q に依存しない！
  - おまけ: 同記録の立ち上がり部から、J固定で実効導通率γを逆算（構成Aの物理）

Caveats / 注意:
  - 反射テープの分だけ J は素のプロペラより大きい（数%オーダの過大の可能性）
  - C_Q は旧プロペラの実測値 → J の絶対値には系統誤差が乗り得る
    （J/C_Q と τ_z はこの影響を受けない）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import least_squares

WFM = "/Users/kouhei/tmp/github/multicopter_introduction/data/stampfly_prop_round_test.wfm"
CQ = 9.72e-11
CT = 1.0e-8
KT = KE = 5.35e-4
R_W = 0.593
M, G = 0.0368, 9.81
OMEGA_HOVER = np.sqrt(M * G / 4 / CT)

# ---------- 1. load & dip detection / 読み込みとディップ検出 ----------
import RigolWFM.wfm as rigol

w = rigol.Wfm.from_file(WFM, "1000Z")
ch = w.channels[0]
v = np.asarray(ch.volts)
t = np.asarray(ch.times)
t = t - t[0]
dt_s = t[1] - t[0]

TH = 4.05
below = v < TH
d = np.diff(below.astype(int))
starts = np.where(d == 1)[0] + 1
ends = np.where(d == -1)[0] + 1
if below[0]: starts = np.r_[0, starts]
if below[-1]: ends = np.r_[ends, len(v)]
dip_t, dip_v = [], []
for a, b in zip(starts, ends):
    if b - a < 8:          # <4µs: noise glitch / ノイズ除去
        continue
    i = a + np.argmin(v[a:b])
    dip_t.append(t[i]); dip_v.append(v[i])
dip_t = np.array(dip_t); dip_v = np.array(dip_v)
print(f"dips detected: {len(dip_t)} over {dip_t[0]:.2f}..{dip_t[-1]:.2f} s")

# remove residual glitches: drop a dip if its spacing to the previous is
# <30% of the local median spacing (keep the deeper one)
# 局所中央値の30%未満の異常間隔は浅い方を除去
keep = np.ones(len(dip_t), dtype=bool)
for i in range(1, len(dip_t)):
    if not keep[i - 1]:
        continue
    lo = max(0, i - 8)
    med = np.median(np.diff(dip_t[lo:i + 1])) if i - lo >= 2 else None
    if med and (dip_t[i] - dip_t[i - 1]) < 0.3 * med:
        if dip_v[i] < dip_v[i - 1]:
            keep[i - 1] = False
        else:
            keep[i] = False
dip_t, dip_v = dip_t[keep], dip_v[keep]
print(f"after glitch removal: {len(dip_t)}")

# ---------- 2. omega(t): 4-interval revolution period / 4間隔=1回転 ----------
# blade asymmetry cancels over any 4 consecutive blade passages
rev_dt = dip_t[4:] - dip_t[:-4]
om = 2 * np.pi / rev_dt
tm = (dip_t[4:] + dip_t[:-4]) / 2

# steady plateau & cutoff detection / 定常値とカット時刻
w_steady = np.median(om[(tm > tm[0] + 0.5) & (om > 0.9 * np.max(om))])
plateau = tm[om > 0.98 * w_steady]
t_cut = plateau[-1]
print(f"steady omega (drive): {w_steady:.0f} rad/s ({w_steady*60/2/np.pi:.0f} rpm)")
print(f"cutoff detected at t = {t_cut:.3f} s")

coast = (tm > t_cut + 0.01)
tc, oc = tm[coast], om[coast]
print(f"coast samples: {len(oc)}, omega {oc.max():.0f} -> {oc.min():.0f} rad/s "
      f"over {tc[-1]-tc[0]:.2f} s")

# ---------- 3. windowed 1/omega slopes / 窓別の1/ω傾き（摩擦の診断） ----------
print("\n1/ω local slope a = C_Q/J by omega band (pure-aero check):")
print("  （純空力なら全帯域で一定。低速で増えるなら摩擦の混入）")
for wlo, whi in ((0.7, 1.0), (0.5, 0.7), (0.3, 0.5), (0.15, 0.3)):
    m = (oc > wlo * oc.max()) & (oc <= whi * oc.max())
    if m.sum() < 8:
        continue
    p = np.polyfit(tc[m], 1.0 / oc[m], 1)
    print(f"  ω/ω0 {wlo:.2f}-{whi:.2f}: a = {p[0]:.3e} 1/rad  -> J/CQ = {1/p[0]:.1f} s·rad, "
          f"J = {CQ/p[0]:.3e} kg·m²")

# ---------- 4. full friction model fit / 3項モデルの全域フィット ----------
def sim_coast(a, b, c, om0, tq):
    dtq = 2e-4
    tt = np.arange(tq[0], tq[-1] + dtq, dtq)
    o = np.zeros(len(tt)); o[0] = om0
    for i in range(1, len(tt)):
        x = o[i - 1]
        f = lambda y: -(a * y * y + b * y + c) if y > 0 else 0.0
        k1 = f(x); k2 = f(x + 0.5 * dtq * k1); k3 = f(x + 0.5 * dtq * k2); k4 = f(x + dtq * k3)
        o[i] = max(0.0, x + dtq / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.interp(tq, tt, o)

def resid(p):
    a, b, c, om0 = p
    return sim_coast(a, b, c, om0, tc) - oc

fit = least_squares(resid, x0=[2e-3, 0.1, 10.0, oc.max()],
                    bounds=([1e-4, 0.0, 0.0, oc.max() * 0.9],
                            [1e-1, 10.0, 1e4, oc.max() * 1.1]),
                    xtol=1e-14, ftol=1e-14)
a_, b_, c_, om0_ = fit.x
rms = np.sqrt(np.mean(fit.fun**2))
print("\n== full coast fit: ω̇ = −a·ω² − b·ω − c ==")
print(f"  a = C_Q/J = {a_:.4e} 1/rad   b = B/J = {b_:.3e} 1/s   c = τ_c/J = {c_:.2f} rad/s²")
print(f"  RMS = {rms:.1f} rad/s ({100*rms/oc.max():.1f}%)")
print(f"  J/C_Q = {1/a_:.1f} s·rad")
print(f"  ★ J = C_Q/a = {CQ/a_:.3e} kg·m²   (C_Q=9.72e-11 使用)")
print(f"  B  = {b_*CQ/a_:.2e} N·m·s/rad,  τ_c = {c_*CQ/a_:.2e} N·m")
print(f"  ★ ヨー零点 τ_z(ホバ, C_Q非依存) = 1/(2·a·ω_hover) = {1/(2*a_*OMEGA_HOVER)*1e3:.1f} ms")

# aero vs friction shares at the coast start / コースト開始点での減衰内訳
w0 = oc.max()
tot = a_ * w0**2 + b_ * w0 + c_
print(f"  減衰内訳 @ω={w0:.0f}: 空力 {100*a_*w0**2/tot:.0f}% / 粘性 {100*b_*w0/tot:.0f}% / クーロン {100*c_/tot:.0f}%")

# ---------- 5. bonus: spin-up conduction fraction with J fixed ----------
# 立ち上がり部（同記録）から、J を固定して実効導通率 γ を逆算
J_hat = CQ / a_
B_FULL = KT * KE / R_W
spin = (tm > t_cut - 2.2) & (tm < t_cut - 0.05) & (om > 20)
ts_, os_ = tm[spin], om[spin]

def sim_spin(A, gamma, t0):
    dtq = 2e-4
    tt = np.arange(t0, ts_[-1] + dtq, dtq)
    o = np.zeros(len(tt))
    for i in range(1, len(tt)):
        x = o[i - 1]
        f = lambda y: (A - gamma * B_FULL * y - CQ * y * y) / J_hat
        k1 = f(x); k2 = f(x + 0.5 * dtq * k1); k3 = f(x + 0.5 * dtq * k2); k4 = f(x + dtq * k3)
        o[i] = x + dtq / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.interp(ts_, tt, o, left=0.0)

Ag = B_FULL * w_steady + CQ * w_steady**2
r2 = least_squares(lambda p: sim_spin(p[0], p[1], p[2]) - os_,
                   x0=[Ag * 0.5, 0.3, ts_[0] - 0.05],
                   bounds=([Ag * 0.05, 0.0, ts_[0] - 0.5], [Ag * 3.0, 1.0, ts_[0] + 0.5]))
print(f"\n== bonus: spin-up with J fixed to coast-down value ==")
print(f"  実効導通率 γ = {r2.x[1]:.3f}  (RMS {np.sqrt(np.mean(r2.fun**2)):.1f} rad/s)")

# ---------- 6. plots ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
axes[0].plot(tm, om, ".", ms=2)
axes[0].axvline(t_cut, color="r", ls="--", lw=1, label=f"cutoff {t_cut:.2f}s")
axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("ω [rad/s]")
axes[0].set_title("full record: spin-up, drive, coast-down"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(tc, 1 / oc, ".", ms=3, label="measured 1/ω")
tq = np.linspace(tc[0], tc[-1], 400)
axes[1].plot(tq, 1 / np.maximum(sim_coast(a_, b_, c_, om0_, tq), 1), "r-", lw=1.5, label="3-term fit")
axes[1].plot(tq, 1 / oc.max() + a_ * (tq - tc[0]), "g--", lw=1.2, label="pure aero (slope a)")
axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("1/ω [s/rad]")
axes[1].set_title("coast-down: 1/ω linearity"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(ts_, os_, ".", ms=2, label="measured")
axes[2].plot(ts_, sim_spin(*r2.x), "r-", lw=1.5, label=f"fit γ={r2.x[1]:.2f}")
axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("ω [rad/s]")
axes[2].set_title("spin-up (J fixed from coast)"); axes[2].legend(); axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/coastdown_id.png", dpi=130)
print("\nplot saved: notes/calc/coastdown_id.png")
