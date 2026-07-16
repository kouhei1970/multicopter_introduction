#!/usr/bin/env python3
"""How close can the outer (angle) loop get to the inner (rate) loop bandwidth?
カスケード制御の内外帯域比を StampFly の実モデルで実証する（Q5-4 の宿題）。

Model / モデル（第4章までの同定値・workshop lesson_06 のロール軸）:
  レートプラント: G(s) = K·e^(−Ls) / (s(τs+1)),  K=102 rad/s²/入力, τ=0.02 s
  むだ時間 L: 0（理想）/ 5ms（センサ+処理の目安）/ 12ms（acro同定の実測級）
  内側: レートP制御（2次系設計 ζ=0.7 → Kp_r = 1/(4ζ²Kτ) = 0.25）
        → 内側帯域 ω_in ≈ ωn = √(Kp_r·K/τ) ≈ 36 rad/s
  外側: 角度P制御 Kp_a（理想なら外側交差 ≈ Kp_a）

Experiment / 実験:
  帯域比 ratio = Kp_a/ω_in を 0.1〜1.0 で掃引し、角度ステップ応答の
  オーバーシュートと2%整定時間を計測。「1/3前後が目安」という経験則が
  このモデルでどう見えるかを定量化する（理想と現実=むだ時間の対比つき）。
"""

import numpy as np

K = 102.0        # effective plant gain [rad/s^2 per unit input]
TAU = 0.02       # motor lag [s]
ZETA = 0.7
KP_R = 1.0 / (4 * ZETA**2 * K * TAU)          # rate P gain
W_IN = np.sqrt(KP_R * K / TAU)                # inner natural freq ≈ bandwidth
DT = 2e-4
T_END = 4.0


def simulate(kp_a, L):
    """cascade step response with dead time L / むだ時間L付きカスケード応答"""
    n = int(T_END / DT)
    nd = max(0, int(round(L / DT)))
    ubuf = np.zeros(nd + 1)                    # dead-time ring buffer
    m = r = th = 0.0
    th_ref = 1.0
    out = np.empty(n)
    for i in range(n):
        r_cmd = kp_a * (th_ref - th)           # outer: angle P
        u = KP_R * (r_cmd - r)                 # inner: rate P
        if nd:
            ubuf[1:] = ubuf[:-1]
            ubuf[0] = u
            u_eff = ubuf[-1]
        else:
            u_eff = u
        m += (u_eff - m) / TAU * DT            # motor first-order lag
        r += K * m * DT                        # rate = ∫ K·m
        th += r * DT
        out[i] = th
    return out


def metrics(y):
    peak = np.nanmax(y)
    overshoot = max(0.0, (peak - 1.0) * 100)
    outside = np.abs(y - 1.0) > 0.02
    settle = (np.max(np.nonzero(outside)) + 1) * DT if outside.any() else 0.0
    return overshoot, settle


print(f"Kp_r = {KP_R:.3f}, 内側帯域 ω_in ≈ {W_IN:.1f} rad/s\n")
ratios = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0])
results = {}
for L in (0.0, 0.005, 0.012):
    rows = []
    for rr in ratios:
        y = simulate(rr * W_IN, L)
        ov, st = metrics(y)
        ok = np.isfinite(y[-1]) and abs(y[-1] - 1) < 0.5
        rows.append((rr, ov, st, ok))
    results[L] = rows
    print(f"-- むだ時間 L = {L*1e3:.0f} ms --")
    print("  比率   OS[%]   整定[s]")
    for rr, ov, st, ok in rows:
        flag = "" if ok and ov < 500 else "  ← 破綻"
        print(f"  {rr:4.2f}  {min(ov,999):6.1f}  {st:6.2f}{flag}")

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
jp = None
for cand in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP"):
    if any(cand in f.name for f in fm.fontManager.ttflist):
        jp = cand
        break
if jp:
    plt.rcParams["font.family"] = jp

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
colors = {0.0: "tab:green", 0.005: "tab:blue", 0.012: "tab:red"}
labels = {0.0: "L=0（理想）", 0.005: "L=5ms", 0.012: "L=12ms"}
for L, rows in results.items():
    rs = [r[0] for r in rows]
    ov = [min(r[1], 200) for r in rows]
    st = [r[2] for r in rows]
    axes[0].plot(rs, ov, "o-", color=colors[L], label=labels[L])
    axes[1].plot(rs, st, "o-", color=colors[L], label=labels[L])
axes[0].axvline(1/3, color="gray", ls=":", lw=1)
axes[0].text(0.34, 120, "1/3", color="gray")
axes[0].set_xlabel("帯域比 ω_out/ω_in")
axes[0].set_ylabel("オーバーシュート [%]")
axes[0].set_title("角度ステップのオーバーシュート")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axvline(1/3, color="gray", ls=":", lw=1)
axes[1].set_xlabel("帯域比 ω_out/ω_in")
axes[1].set_ylabel("2%整定時間 [s]")
axes[1].set_title("整定時間（攻めすぎると逆に遅くなる）")
axes[1].legend(); axes[1].grid(alpha=0.3)
tvec = np.arange(int(T_END / DT)) * DT
for rr, lsty in [(0.2, "-"), (1/3, "-"), (0.5, "--"), (0.8, ":")]:
    y = simulate(rr * W_IN, 0.012)
    axes[2].plot(tvec, y, lsty, label=f"比率 {rr:.2f}")
axes[2].axhline(1, color="gray", lw=0.7)
axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("角度")
axes[2].set_title("ステップ応答（L=12ms、現実条件。比率0.8は発散）")
axes[2].legend(); axes[2].grid(alpha=0.3)
axes[2].set_xlim(0, 2.5)
axes[2].set_ylim(-0.1, 2.0)
fig.tight_layout()
fig.savefig("notes/calc/cascade_bandwidth_study.png", dpi=130)
print("\nplot saved: notes/calc/cascade_bandwidth_study.png")
