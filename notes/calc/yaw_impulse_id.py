#!/usr/bin/env python3
"""FIR identification: motor-duty differential -> gyro, yaw vs roll.
ミキサ出力の差動Duty（プラント実入力）→ジャイロのFIR同定でヨーの初期応答符号を判定。

Background / 背景:
  ヨー指令（スティック）はランプ状で蹴り返しを励起できない。代わりに
  ctrl_ref.motor_duty[4]（400Hz記録）から軸別の差動入力を作り、gyro への
  インパルス応答を最小二乗FIRで直接同定する。フィードバックが注入する
  高周波成分が励起源。閉ループ直接同定なのでバイアスはありうるが、
  「初期応答の符号」という粗い判定には十分。方法の妥当性検証として、
  零点を持たないはずのロール軸を同一手法で並走させる（正の応答が出るはず）。

Axis inputs (M1:FR CCW, M2:RR CW, M3:RL CCW, M4:FL CW):
  u_roll  = (-d1 - d2 + d3 + d4)   left up  -> +roll(右下がり)
  u_pitch = (+d1 - d2 - d3 + d4)   front up -> +pitch(機首上げ)
  u_yaw   = (+d1 - d2 + d3 - d4)   CCW up   -> +yaw(機首右)

Judgement / 判定:
  ヨーFIRの累積（ステップ応答）が最初に負へ振れる → RHP零点（非最小位相）
  最初から正に立ち上がる → 進み零点（最小位相）
"""

import json
import numpy as np

LOGS = [
    "/Users/kouhei/tmp/github/stampfly_ecosystem/logs/stampfly_udp_20260627T164611.jsonl",
    "/Users/kouhei/tmp/github/stampfly_ecosystem/logs/stampfly_udp_20260627T165713.jsonl",
]
DT = 0.0025          # 400 Hz grid
NTAP = 120           # FIR length: 0.3 s
LAMBDA = 1e-4        # ridge regularization (relative)


def load(path):
    tc, du = [], []
    tg, gy = [], []
    with open(path) as f:
        for line in f:
            if '"ctrl_ref"' in line:
                d = json.loads(line)
                du.append(d["motor_duty"]); tc.append(d["ts"] * 1e-6)
            elif '"id":"imu"' in line:
                d = json.loads(line)
                gy.append(d["gyro"]); tg.append(d["ts"] * 1e-6)
    tc = np.array(tc); du = np.array(du)
    tg = np.array(tg); gy = np.array(gy)
    i = np.argsort(tc); tc, du = tc[i], du[i]
    i = np.argsort(tg); tg, gy = tg[i], gy[i]
    return tc, du, tg, gy


def largest_flight_block(tc, du):
    """largest contiguous in-flight span / 最長の飛行区間"""
    ok = np.all((du > 0.2) & (du < 0.95), axis=1)
    d = np.diff(ok.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if ok[0]: starts = np.r_[0, starts]
    if ok[-1]: ends = np.r_[ends, len(ok)]
    k = np.argmax(ends - starts)
    return tc[starts[k]], tc[ends[k] - 1]


def detrend(x, win):
    """subtract moving average / 移動平均を引いて低周波トレンド除去"""
    kernel = np.ones(win) / win
    trend = np.convolve(x, kernel, mode="same")
    return x - trend


def fir_fit(u, y, ntap, lam):
    """ridge least-squares FIR: y[k] = sum h[i] u[k-i]"""
    N = len(u)
    U = np.zeros((N - ntap, ntap))
    for i in range(ntap):
        U[:, i] = u[ntap - i:N - i]
    yy = y[ntap:]
    A = U.T @ U
    A += lam * np.trace(A) / ntap * np.eye(ntap)
    h = np.linalg.solve(A, U.T @ yy)
    # fit quality / 当てはまり
    r2 = 1 - np.var(yy - U @ h) / np.var(yy)
    return h, r2


results = {}
for axis in ("roll", "yaw"):
    results[axis] = []

for path in LOGS:
    tc, du, tg, gy = load(path)
    t0, t1 = largest_flight_block(tc, du)
    grid = np.arange(t0 + 1.0, t1 - 1.0, DT)
    d = [np.interp(grid, tc, du[:, i]) for i in range(4)]
    u_roll = -d[0] - d[1] + d[2] + d[3]
    u_yaw = d[0] - d[1] + d[2] - d[3]
    gx = np.interp(grid, tg, gy[:, 0])
    gz = np.interp(grid, tg, gy[:, 2])
    win = int(0.5 / DT)
    for axis, u, y in (("roll", u_roll, gx), ("yaw", u_yaw, gz)):
        h, r2 = fir_fit(detrend(u, win), detrend(y, win), NTAP, LAMBDA)
        results[axis].append(h)
        print(f"{path.split('/')[-1]} {axis:5s}: flight {t1-t0:.0f}s, FIR R²={r2:.3f}")

# ---- summarize & plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lags = np.arange(NTAP) * DT * 1e3   # ms
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, axis in enumerate(("roll", "yaw")):
    hs = results[axis]
    hm = np.mean(hs, axis=0)
    step = np.cumsum(hm) * DT
    ax = axes[0][col]
    for h in hs:
        ax.plot(lags, h, alpha=0.4, lw=1)
    ax.plot(lags, hm, "k", lw=2)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title(f"{axis}: impulse response (duty diff → gyro)")
    ax.set_xlabel("lag [ms]"); ax.grid(alpha=0.3)
    ax = axes[1][col]
    ax.plot(lags, np.cumsum(np.array(hs).T * DT, axis=0), alpha=0.5, lw=1)
    ax.plot(lags, step, "k", lw=2)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title(f"{axis}: step response (cumulative)")
    ax.set_xlabel("lag [ms]"); ax.grid(alpha=0.3)
    # quantify early lags / 初期ラグの定量
    print(f"\n{axis} mean impulse response by lag window:")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80)):
        m = (lags >= lo) & (lags < hi)
        print(f"  {lo:3d}-{hi:3d} ms: {hm[m].mean():+8.3f}")
    s0 = step[(lags >= 0) & (lags < 60)]
    print(f"  step-response min over 0-60ms: {s0.min():+.4f} "
          f"(at {lags[(lags>=0)&(lags<60)][s0.argmin()]:.1f} ms), "
          f"value at 60ms: {step[lags<=60][-1]:+.4f}")

fig.tight_layout()
fig.savefig("/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/yaw_impulse_id.png", dpi=130)
print("\nplot saved: notes/calc/yaw_impulse_id.png")
