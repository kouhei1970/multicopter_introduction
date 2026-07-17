#!/usr/bin/env python3
"""Generate Japanese-labeled figures for the appendix pages from real data.
付録ページ用の図を実測データから生成する（日本語ラベル・教材用）。

Outputs -> assets/img/appendix/
  a1_waveform_zoom.png : フォトインタラプタ生波形（ブレード4ディップ+テープの深いディップ）
  a2_omega.png         : ω(t) 全記録（立ち上がり→定常→コーストダウン）
  a3_inverse_omega.png : 1/ω の直線化とフィット
  c1_overview.png      : Ke測定 2ch 全景（ドレイン電圧のEMF降下と停止後の電池電圧）
  c2_ke_fit.png        : V–ω 直線フィット（傾き = −Ke）

Data: data/stampfly_prop_round_test.wfm（付録A、時刻軸のみ使用・振幅は生値）
      data/coastdown_scpi_refetch.npz（付録C、SCPI公式換算の電圧）
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

for cand in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP"):
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break

OUT = "assets/img/appendix"
os.makedirs(OUT, exist_ok=True)


def tape_pulses(t, v, hi_off=0.15, lo_frac=0.55):
    """tape pulse = deepest dip among neighbours / テープ=両隣より深いディップ"""
    base = np.percentile(v, 95)
    vmin = v.min()
    lo = vmin + lo_frac * (base - vmin)
    hi = base - hi_off
    s = np.zeros(len(v), dtype=np.int8)
    s[v > hi] = 1
    s[v < lo] = -1
    nz = np.flatnonzero(s)
    sv = s[nz]
    chg = np.flatnonzero(np.diff(sv) != 0)
    starts = nz[chg + 1][sv[chg + 1] == -1]
    ends_i = nz[chg + 1][sv[chg + 1] == 1]
    dips_t, dips_d = [], []
    j = 0
    for a in starts:
        while j < len(ends_i) and ends_i[j] <= a:
            j += 1
        b = ends_i[j] if j < len(ends_i) else len(v)
        i = a + int(np.argmin(v[a:b]))
        dips_t.append(t[i]); dips_d.append(v[i])
    dips_t = np.array(dips_t); dips_d = np.array(dips_d)
    if len(dips_t) < 3:
        return dips_t
    tape = np.zeros(len(dips_t), dtype=bool)
    tape[1:-1] = (dips_d[1:-1] < dips_d[:-2]) & (dips_d[1:-1] < dips_d[2:])
    edges = dips_t[tape]
    keep = np.ones(len(edges), dtype=bool)
    for i in range(1, len(edges)):
        lo_i = max(0, i - 8)
        if i - lo_i >= 2:
            med = np.median(np.diff(edges[lo_i:i + 1]))
            if (edges[i] - edges[i - 1]) < 0.5 * med:
                keep[i] = False
    return edges[keep]


# ================= 付録A: コーストダウン =================
import RigolWFM.wfm as rigol
w = rigol.Wfm.from_file("data/stampfly_prop_round_test.wfm", "1000Z")
ch = w.channels[0]
tA = np.asarray(ch.times); tA = tA - tA[0]
vA = np.asarray(ch.volts)

edges = tape_pulses(tA, vA)
om = 2 * np.pi / np.diff(edges)
tm = (edges[1:] + edges[:-1]) / 2
w_steady = np.median(om[om > 0.9 * om.max()])
t_cut = tm[om > 0.98 * w_steady][-1]
print(f"A: pulses={len(edges)}, steady={w_steady:.0f} rad/s, cutoff={t_cut:.2f}s")

# --- A1: 波形ズーム（定常中の約2.5回転分）---
i0 = np.searchsorted(tA, t_cut - 0.02)
i1 = np.searchsorted(tA, t_cut - 0.02 + 2.6 * (2 * np.pi / w_steady))
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot((tA[i0:i1] - tA[i0]) * 1e3, vA[i0:i1], lw=1.0)
ax.set_xlabel("時間 [ms]")
ax.set_ylabel("フォトインタラプタ出力 [V]（生値）")
ax.set_title("生波形: 1回転に4枚のブレードが4つのディップを作る。テープのディップだけ深い")
seg_t = tA[i0:i1] - tA[i0]
seg_v = vA[i0:i1]
idx = int(np.argmin(seg_v))                      # 最深点 = テープ / deepest = tape
ax.annotate("反射テープ（1回転に1回・いちばん深い）",
            xy=(seg_t[idx] * 1e3, seg_v[idx]),
            xytext=(seg_t[idx] * 1e3 + 1.6, seg_v.min() + 0.12),
            arrowprops=dict(arrowstyle="->", color="#c2373b"), color="#c2373b", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/a1_waveform_zoom.png", dpi=130)
plt.close(fig)

# --- A2: ω(t) ---
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(tm, om, ".", ms=3)
ax.axvline(t_cut, color="#c2373b", ls="--", lw=1.2)
ax.text(t_cut + 0.03, om.max() * 0.9, "電源カット", color="#c2373b")
ax.set_xlabel("時間 [s]")
ax.set_ylabel("回転数 ω [rad/s]")
ax.set_title("テープパルスの間隔から復元した ω(t): 立ち上がり → 定常 → 惰性減速")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/a2_omega.png", dpi=130)
plt.close(fig)

# --- A3: 1/ω と 3項フィット（ω̇=−aω²−bω−c、クーロン摩擦込み） ---
from scipy.optimize import least_squares
coast = tm > t_cut + 0.01
tc_, oc_ = tm[coast] - t_cut, om[coast]

def sim_coast(a, b, c, om0, tq):
    dtq = 2e-4
    tt = np.arange(tq[0], tq[-1] + dtq, dtq)
    o = np.zeros(len(tt)); o[0] = om0
    for i in range(1, len(tt)):
        xx = o[i - 1]
        f = lambda yv: -(a * yv * yv + b * yv + c) if yv > 0 else 0.0
        k1 = f(xx); k2 = f(xx + 0.5 * dtq * k1); k3 = f(xx + 0.5 * dtq * k2); k4 = f(xx + dtq * k3)
        o[i] = max(0.0, xx + dtq / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.interp(tq, tt, o)

fit3 = least_squares(lambda pz: sim_coast(pz[0], pz[1], pz[2], pz[3], tc_) - oc_,
                     x0=[2e-3, 0.1, 10.0, oc_.max()],
                     bounds=([1e-4, 0.0, 0.0, oc_.max() * 0.9],
                             [1e-1, 10.0, 1e4, oc_.max() * 1.1]),
                     xtol=1e-14, ftol=1e-14)
a3_, b3_, c3_, om03_ = fit3.x
print(f"A3: 3-term fit a={a3_:.4e} -> J/C_Q={1/a3_:.1f} s·rad, b={b3_:.2e}, c={c3_:.1f}")

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(tc_, 1 / oc_, ".", ms=4, label="実測 1/ω")
tq = np.linspace(tc_[0], tc_[-1], 200)
ax.plot(tq, 1 / np.maximum(sim_coast(a3_, b3_, c3_, om03_, tq), 1), "r-", lw=1.5,
        label=f"3項フィット ω̇=−aω²−bω−c → J/C_Q = 1/a = {1/a3_:.0f} s·rad")
ax.plot(tq, 1 / om03_ + a3_ * (tq - tq[0]), "g--", lw=1.2,
        label="純空力だけの直線（傾き a）——実測はここから上に反る")
ax.set_xlabel("電源カットからの時間 [s]")
ax.set_ylabel("1/ω [s/rad]")
ax.set_title("空力だけなら 1/ω は直線。上への反りがクーロン摩擦の署名")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/a3_inverse_omega.png", dpi=130)
plt.close(fig)

# ================= 付録C: Ke =================
z = np.load("data/coastdown_scpi_refetch.npz")
tC = z["t"]; tC = tC - tC[0]
vp = z["ch1_v"]          # photointerrupter
vd = z["ch2_v"]          # drain voltage (SCPI official volts)

edgesC = tape_pulses(tC, vp)
omC = 2 * np.pi / np.diff(edgesC)
tmC = (edgesC[1:] + edgesC[:-1]) / 2
wsC = np.median(omC[omC > 0.9 * omC.max()])
t_cutC = tmC[omC > 0.98 * wsC][-1]
print(f"C: pulses={len(edgesC)}, steady={wsC:.0f} rad/s, cutoff={t_cutC:.2f}s")

# --- C1: 全景 ---
fig, ax = plt.subplots(figsize=(9, 3.8))
dec = slice(None, None, 20)
ax.plot(tC[dec], vd[dec], lw=0.4)
ax.set_xlabel("時間 [s]")
ax.set_ylabel("モータ端子（ドレイン）電圧 [V]")
ax.set_title("全景: 回転中はPWMで振動、電源カット後は V = V_BAT − Ke·ω がそのまま見える")
ax.annotate("電源カット\n（ここから電流ゼロ）", xy=(t_cutC, 3.6), xytext=(t_cutC - 1.9, 1.2),
            arrowprops=dict(arrowstyle="->", color="#c2373b"), color="#c2373b", fontsize=9)
tail = tC > edgesC[-1] + 0.3
vbat = vd[tail].mean()
ax.annotate(f"停止後の直流レベル = 電池電圧 {vbat:.2f} V",
            xy=(tC[-1] - 0.3, vbat), xytext=(tC[-1] - 2.6, 6.3),
            arrowprops=dict(arrowstyle="->", color="#2e8b57"), color="#2e8b57", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/c1_overview.png", dpi=130)
plt.close(fig)

# --- C2: V–ω 直線 ---
vbar = np.array([vd[(tC >= a) & (tC < b)].mean() for a, b in zip(edgesC[:-1], edgesC[1:])])
coastC = tmC > t_cutC + 0.01
xw, yv = omC[coastC], vbar[coastC]
A_ = np.vstack([xw, np.ones_like(xw)]).T
(slopeC, icptC), *_ = np.linalg.lstsq(A_, yv, rcond=None)
yhat = A_ @ [slopeC, icptC]
r2 = 1 - np.sum((yv - yhat) ** 2) / np.sum((yv - yv.mean()) ** 2)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(xw, yv, ".", ms=5, label="コースト中の実測（回転1周ごとに平均）")
xs = np.linspace(0, xw.max(), 50)
ax.plot(xs, icptC + slopeC * xs, "r-", lw=1.5,
        label=f"直線フィット: 傾き = −Ke = {slopeC:.3e} → Ke = {-slopeC:.3e} V·s/rad")
ax.axhline(vbat, color="#2e8b57", ls=":", label=f"停止後の電池電圧 {vbat:.3f} V（切片 {icptC:.3f} V と一致）")
ax.set_xlabel("回転数 ω [rad/s]")
ax.set_ylabel("端子電圧 [V]")
ax.set_title(f"V = V_BAT − Ke·ω の直線（R² = {r2:.4f}）— 電流計もトルク計も不要")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/c2_ke_fit.png", dpi=130)
plt.close(fig)
print(f"C: Ke={-slopeC:.4e}, icpt={icptC:.3f}, V_BAT={vbat:.3f}, R2={r2:.5f}")

# ================= 付録B: 翼弦分布（日本語版） =================
# prop_planform 相当を日本語ラベルで作り直す（画素直接積分の中間生成物）
print("appendix figures saved to", OUT)
