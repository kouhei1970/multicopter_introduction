#!/usr/bin/env python3
"""Motor+prop identification from the photointerrupter transient (NewFile3.wfm).
フォトインタラプタ過渡測定（NewFile3.wfm）からのモータ+プロペラ同定。

Data / データ:
  rigol-wfm-viewer/samples/NewFile3.wfm — DS1054Z, 5 MSa/s, 2.4 s, 12M points.
  反射式フォトインタラプタのパルス列。静止 → 定常回転への立ち上がり。

Pulse structure found by inspection / 事前検査で判明したパルス構造:
  - 定常部の間隔は完全に均一（~5.13 ms, mod-4 パターン無し）
    → 基本は 1 パルス = 1 回転（README の「約11,700rpm」= 1/5.13ms×60 と整合）
  - 立ち上がり中のみ「短・長」交互の余分な反射パルスが混入（和は滑らかに変化）
    → 回転周期の連続性を使った後ろ向き併合で除去する

Identification / 同定:
  モデル: J·dω/dt = A − B_emf·ω − C_Q·ω²,  A = Kt·V·D/R（入力・未知）
  固定値（モータドライバ論文）: B_emf = Kt·Ke/R = 4.827e-7, C_Q = 9.72e-11
  推定値: J（と A, t0）を非線形最小二乗でフィット
  ※ C_Q は「旧プロペラ」の実測値。本測定のプロペラと違う場合は J に系統誤差が乗る。

Outputs: fitted J, time constants, comparison with the two J candidates
  (firmware 2.01e-8 / paper 5.31e-8), and a PNG overlay plot.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import least_squares

WFM = "/Users/kouhei/tmp/github/rigol-wfm-viewer/samples/NewFile3.wfm"
B_EMF = 5.35e-4 * 5.35e-4 / 0.593   # Kt*Ke/R [N·m·s/rad]
CQ = 9.72e-11                        # [N·m·s²/rad²]

# ---------- 1. load & edge detection / 読み込みとエッジ検出 ----------
import RigolWFM.wfm as rigol

w = rigol.Wfm.from_file(WFM, "1000Z")
ch = w.channels[0]
v = np.asarray(ch.volts)
t = np.asarray(ch.times)
t = t - t[0]

HI, LO = 4.05, 3.55                  # hysteresis thresholds [V]
s = np.zeros(len(v), dtype=np.int8)
s[v > HI] = 1
s[v < LO] = -1
nz = np.flatnonzero(s)
sv = s[nz]
chg = np.flatnonzero(np.diff(sv) != 0)
fall = nz[chg + 1][sv[chg + 1] == -1]
edges = t[fall]
print(f"falling edges: {len(edges)}, span {edges[0]:.3f}..{edges[-1]:.3f} s")

# ---------- 2. merge spurious pulses / 余分パルスの併合 ----------
# Work backwards from the clean steady-state end, tracking the rev period.
# 綺麗な定常端から後ろ向きに、回転周期の連続性で間隔を併合する。
iv = np.diff(edges)[::-1]            # reversed intervals
P = np.median(iv[:20])               # steady period estimate
periods_rev, t_end_rev = [], []
te = edges[-1]
acc = 0.0
for dt_i in iv:
    acc += dt_i
    if acc >= 0.6 * P:               # accepted as one revolution
        periods_rev.append(acc)
        t_end_rev.append(te)
        te -= acc
        P = acc
        acc = 0.0
    # else: spurious extra pulse -> keep accumulating / 余分パルス: 併合継続
periods = np.array(periods_rev[::-1])
t_mid = np.array(t_end_rev[::-1]) - periods / 2
omega = 2 * np.pi / periods
print(f"revolutions reconstructed: {len(omega)}")
print(f"steady omega (last 0.2s): {omega[t_mid > t_mid[-1]-0.2].mean():.1f} rad/s "
      f"({omega[t_mid > t_mid[-1]-0.2].mean()*60/2/np.pi:.0f} rpm)")

# drop the first few revolutions (start-up pulse chaos) / 最初の数回転は捨てる
omega_f, t_f = omega[3:], t_mid[3:]

# ---------- 3. fits / フィット ----------
def sim(J, A, t0, tq):
    """integrate J·ω̇ = A − B_emf·ω − CQ·ω² from ω(t0)=0 / RK4積分"""
    dt = 1e-4
    tt = np.arange(t0, tq[-1] + dt, dt)
    om = np.zeros(len(tt))
    for i in range(1, len(tt)):
        o = om[i - 1]
        f = lambda x: (A - B_EMF * x - CQ * x * x) / J
        k1 = f(o); k2 = f(o + 0.5 * dt * k1); k3 = f(o + 0.5 * dt * k2); k4 = f(o + dt * k3)
        om[i] = o + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.interp(tq, tt, om, left=0.0)

def resid_phys(p):
    J, A, t0 = p
    return sim(J, A, t0, t_f) - omega_f

# (a) physical fit: J free / 物理モデルフィット
t0_guess = edges[0] - 0.02
w_ss = omega_f[-20:].mean()
A_guess = B_EMF * w_ss + CQ * w_ss**2
fit = least_squares(resid_phys, x0=[4e-8, A_guess, t0_guess],
                    bounds=([1e-9, A_guess * 0.5, t0_guess - 0.3],
                            [5e-7, A_guess * 2.0, edges[0]]),
                    xtol=1e-14, ftol=1e-14)
J_hat, A_hat, t0_hat = fit.x
rms = np.sqrt(np.mean(fit.fun**2))
tau_eff_ss = J_hat / (B_EMF + 2 * CQ * w_ss)
print("\n== physical fit (J free, B_emf & C_Q fixed from paper) ==")
print(f"  J    = {J_hat:.3e} kg·m²")
print(f"  A    = {A_hat:.3e} N·m  (→ 定常 {w_ss:.0f} rad/s と整合)")
print(f"  t0   = {t0_hat:.4f} s (first edge {edges[0]:.4f} s)")
print(f"  RMS residual = {rms:.1f} rad/s ({100*rms/w_ss:.1f}% of ω_ss)")
print(f"  τ_eff at ω_ss = {tau_eff_ss*1e3:.1f} ms")

# (b) candidate comparison / 候補J（2.01e-8, 5.31e-8）でのフィット比較
print("\n== candidate comparison (J fixed, A & t0 refit) ==")
for Jc in (2.01e-8, 5.31e-8):
    r = least_squares(lambda p: sim(Jc, p[0], p[1], t_f) - omega_f,
                      x0=[A_guess, t0_guess],
                      bounds=([A_guess * 0.5, t0_guess - 0.3], [A_guess * 2.0, edges[0]]))
    rmsc = np.sqrt(np.mean(r.fun**2))
    print(f"  J = {Jc:.2e}: RMS = {rmsc:6.1f} rad/s ({100*rmsc/w_ss:.1f}% of ω_ss)")

# ---------- 4. plot / 図 ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_f, omega_f, ".", ms=3, label="measured ω (per revolution)")
    tq = np.linspace(t0_hat, t_f[-1], 800)
    ax.plot(tq, sim(J_hat, A_hat, t0_hat, tq), "-", lw=2,
            label=f"fit: J={J_hat:.2e} kg·m²")
    for Jc, ls in ((2.01e-8, "--"), (5.31e-8, ":")):
        r = least_squares(lambda p: sim(Jc, p[0], p[1], t_f) - omega_f,
                          x0=[A_guess, t0_guess],
                          bounds=([A_guess * 0.5, t0_guess - 0.3], [A_guess * 2.0, edges[0]]))
        ax.plot(tq, sim(Jc, r.x[0], r.x[1], tq), ls, lw=1.5, label=f"J={Jc:.2e} (candidate)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ω [rad/s]")
    ax.set_title("StampFly motor+prop spin-up: photointerrupter measurement vs model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/photointerrupter_fit.png", dpi=130)
    print("\nplot saved: notes/calc/photointerrupter_fit.png")
except ImportError:
    print("\n(matplotlib なし: 図はスキップ)")
