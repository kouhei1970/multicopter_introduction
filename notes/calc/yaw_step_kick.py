#!/usr/bin/env python3
"""Does the StampFly yaw axis show an initial reverse kick (RHP zero)?
StampFly のヨー軸に「蹴り返し」（RHP零点＝非最小位相の初期逆応答）はあるか？

Method / 方法:
  飛行ログ（stampfly_udp jsonl）から
    - rate_ref レコードのヨー指令 r_ref(t)
    - imu レコードのジャイロ z 軸 gyro_z(t)（機体ヨーレート実測）
  を取り出し、ヨー指令のステップ状変化を検出。ステップ前後の gyro_z を
  ステップ符号で正規化して重ね合わせ（アンサンブル平均）、
  立ち上がり直後の符号を見る。
    - 一瞬逆向きに振れる（負の初期応答）→ RHP零点（非最小位相）
    - 最初から指令方向に立ち上がる → 進み零点（最小位相）
  RHP零点の初期逆応答はフィードバックでは消せないため、閉ループログでも判定可能。

Step selection / ステップ選定:
  - 直前 80ms の |r_ref| 平均 < 0.3 rad/s から、40ms 以内に |r_ref| > 1.0 rad/s へ
  - ステップ後 100ms の平均 |r_ref| > 0.8 rad/s（維持されている）
  - 窓内のロール/ピッチ指令 |p_ref|,|q_ref| < 1.0 rad/s（他軸の混入を制限）
"""

import json
import numpy as np

LOGS = [
    "/Users/kouhei/tmp/github/stampfly_ecosystem/logs/stampfly_udp_20260627T164611.jsonl",
    "/Users/kouhei/tmp/github/stampfly_ecosystem/logs/stampfly_udp_20260627T165713.jsonl",
]

PRE, POST = 0.15, 0.35   # window around step [s] / ステップ前後の窓


def load(path):
    tr, rr = [], []      # rate_ref: time, (p_ref,q_ref,r_ref)
    tg, gz = [], []      # gyro z
    with open(path) as f:
        for line in f:
            if '"rate_ref"' in line:
                d = json.loads(line)
                tr.append(d["ts"] * 1e-6)
                rr.append(d["rate_ref"])
            elif '"id":"imu"' in line:
                d = json.loads(line)
                tg.append(d["ts"] * 1e-6)
                gz.append(d["gyro"][2])
    tr = np.array(tr); rr = np.array(rr)
    tg = np.array(tg); gz = np.array(gz)
    # sort (UDP order safety) / 念のため時刻順に
    i = np.argsort(tr); tr, rr = tr[i], rr[i]
    i = np.argsort(tg); tg, gz = tg[i], gz[i]
    return tr, rr, tg, gz


def find_steps(tr, rr):
    """detect step onsets in yaw ref / ヨー指令のステップ開始点を検出"""
    r = rr[:, 2]
    steps = []
    k = 0
    while k < len(tr):
        t0 = tr[k]
        pre = (tr > t0 - 0.08) & (tr < t0)
        post40 = (tr >= t0) & (tr < t0 + 0.04)
        post100 = (tr >= t0) & (tr < t0 + 0.10)
        if pre.sum() > 5 and post100.sum() > 10:
            if (np.abs(r[pre]).mean() < 0.3
                    and np.abs(r[post40]).max() > 1.0
                    and np.abs(r[post100]).mean() > 0.8
                    and np.abs(rr[post100][:, 0]).max() < 1.0
                    and np.abs(rr[post100][:, 1]).max() < 1.0):
                # onset refinement: first sample where |r| exceeds 0.3
                # 立ち上がり点を |r|>0.3 の最初のサンプルに補正
                seg = np.where(post40)[0]
                on = seg[np.argmax(np.abs(r[seg]) > 0.3)]
                sign = np.sign(r[seg][np.abs(r[seg]).argmax()])
                steps.append((tr[on], sign))
                k += int(post100.sum()) + 200   # skip past this step
                continue
        k += 1
    return steps


GRID = np.arange(-PRE, POST, 0.0025)
ens = []
for path in LOGS:
    tr, rr, tg, gz = load(path)
    steps = find_steps(tr, rr)
    print(f"{path.split('/')[-1]}: {len(steps)} yaw steps")
    for t0, sgn in steps:
        m = (tg > t0 - PRE - 0.05) & (tg < t0 + POST + 0.05)
        if m.sum() < 50:
            continue
        y = np.interp(GRID, tg[m] - t0, gz[m])
        base = y[GRID < -0.02].mean()          # pre-step baseline / 直前基準
        ens.append(sgn * (y - base))

ens = np.array(ens)
print(f"total usable steps: {len(ens)}")
mean = ens.mean(axis=0)
sem = ens.std(axis=0) / np.sqrt(len(ens))

# quantify the initial phase / 立ち上がり直後の定量
for lo, hi in ((0.0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08)):
    m = (GRID >= lo) & (GRID < hi)
    print(f"  t = {lo*1e3:3.0f}-{hi*1e3:3.0f} ms: mean = {mean[m].mean():+7.4f} rad/s "
          f"(SEM {sem[m].mean():.4f})")
neg = mean[(GRID >= 0) & (GRID < 0.05)]
print(f"  min over 0-50ms: {neg.min():+.4f} rad/s at t={GRID[(GRID>=0)&(GRID<0.05)][neg.argmin()]*1e3:.1f} ms")

# plot / 図
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
for row in ens:
    ax.plot(GRID * 1e3, row, color="#8ab", alpha=0.25, lw=0.8)
ax.plot(GRID * 1e3, mean, color="#c2373b", lw=2.5, label=f"ensemble mean (n={len(ens)})")
ax.fill_between(GRID * 1e3, mean - 2 * sem, mean + 2 * sem, color="#c2373b", alpha=0.2, label="±2 SEM")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("time from yaw-ref step onset [ms]")
ax.set_ylabel("sign-normalized gyro_z response [rad/s]")
ax.set_title("StampFly yaw step: initial response direction (reverse kick test)")
ax.set_xlim(-100, 300)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/yaw_step_kick.png", dpi=130)
print("plot saved: notes/calc/yaw_step_kick.png")
