#!/usr/bin/env python3
"""Ke identification from a 2-channel coast-down capture (no current sensing).

!! RigolWFM 振幅バグ（2026-07-15 機構特定・決定論）: 本ライブラリは全設定で
!! 「1目盛=20レベル」として復元するが、DS1000Z の文書化された仕様
!! （プログラミングガイド: YINCrement = V/div ÷ 25）は 25 レベル/目盛。
!! → 全振幅が 25/20=1.25 倍。傾き由来量には AMP_FIX=20/25=0.80 を適用済み。
!! オフセット復元にも +0.26V 級の未解明残差あり — 絶対電圧はアンカー必須。
!! 時刻軸は正確。詳細: notes/qa_log.md Q4-20〜21
2ch コーストダウン収録からの逆起電力定数 Ke の同定（電流計測不要）。

Usage / 使い方:
    python3 ke_from_coastdown.py <file.wfm> [--photo CH] [--drain CH]

Measurement setup / 収録条件（先生向けメモ）:
  - いつものコーストダウン（定常まで回す→電源カット→停止まで）を2chで収録
  - CH1: フォトインタラプタ（従来通り）
  - CH2: モータのドレイン側端子電圧（GND基準、DCカップリング、レンジ0〜5V目安）
  - カット後は電流≈0（FET開、100nFの充電電流はµA未満）なので
      V_drain(t) = V_BAT − Ke·ω(t)
    が成り立つ。停止後の直流レベルがそのまま V_BAT（EMF=0, 電流0, IR降下0）。

Analysis / 解析:
  1. フォトインタラプタchからテープパルスを検出し ω(t) を復元（coastdown_id.py と同じ）
  2. ドレインchを各回転周期で平均化（PWMは無いがノイズ低減）
  3. カット後区間で V_drain vs ω を直線フィット → 傾き = −Ke、切片 = V_BAT
  4. 直線性の確認（Ke が定数か）と、停止後DCレベルとの切片一致で検算

判定 / Expected discrimination:
  Ke ≈ 5.35e-4（論文系） vs ≈ 6.125e-4（firmware系）— 14%差は余裕で分解可能。
  確定した Ke と論文の R=0.593Ω（LCR実測・信頼高）で電気ブレーキ KtKe/R が決まり、
  τ_eff の 9.5 vs 17.5 ms 問題が決着する。
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# RigolWFM amplitude fix: library decodes 20 levels/div; documented Rigol
# convention is 25 levels/div (Programming Guide: YINC = V/div ÷ 25)
# ライブラリの振幅復元バグの固定補正（傾き由来量に適用）
AMP_FIX = 20.0 / 25.0


def tape_pulses(t, v, hi_off=0.15, lo_frac=0.55):
    """Detect the tape pulse (deepest dip per revolution).
    テープパルス検出: 全ディップを拾い、隣接より深いものだけをテープと判定。
    （教訓 2026-07-15: 4枚ブレードが全部閾値を跨ぐ個体があり、固定閾値だと
    4パルス/回転を拾って ω を4倍に誤る。テープ=グループ内で最深、で判別する）"""
    base = np.percentile(v, 95)
    vmin = v.min()
    lo = vmin + lo_frac * (base - vmin)   # generous dip threshold / 緩めの閾値
    hi = base - hi_off
    s = np.zeros(len(v), dtype=np.int8)
    s[v > hi] = 1
    s[v < lo] = -1
    nz = np.flatnonzero(s)
    sv = s[nz]
    chg = np.flatnonzero(np.diff(sv) != 0)
    starts = nz[chg + 1][sv[chg + 1] == -1]
    ends_i = nz[chg + 1][sv[chg + 1] == 1]
    # dip list with depth / 各ディップの時刻と深さ
    dips_t, dips_d = [], []
    j = 0
    for a in starts:
        while j < len(ends_i) and ends_i[j] <= a:
            j += 1
        b = ends_i[j] if j < len(ends_i) else len(v)
        i = a + int(np.argmin(v[a:b]))
        dips_t.append(t[i]); dips_d.append(v[i])
    dips_t = np.array(dips_t); dips_d = np.array(dips_d)
    # tape = deeper than BOTH neighbours / 両隣より深い = テープ
    if len(dips_t) < 3:
        return dips_t
    tape = np.zeros(len(dips_t), dtype=bool)
    tape[1:-1] = (dips_d[1:-1] < dips_d[:-2]) & (dips_d[1:-1] < dips_d[2:])
    edges = dips_t[tape]
    # glitch merge / 異常間隔の併合
    keep = np.ones(len(edges), dtype=bool)
    for i in range(1, len(edges)):
        lo_i = max(0, i - 8)
        if i - lo_i >= 2:
            med = np.median(np.diff(edges[lo_i:i + 1]))
            if (edges[i] - edges[i - 1]) < 0.5 * med:
                keep[i] = False
    return edges[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wfm")
    ap.add_argument("--photo", type=int, default=1, help="photointerrupter channel (default 1)")
    ap.add_argument("--drain", type=int, default=2, help="drain-voltage channel (default 2)")
    args = ap.parse_args()

    import RigolWFM.wfm as rigol
    w = rigol.Wfm.from_file(args.wfm, "1000Z")
    chans = {c.channel_number: c for c in w.channels}
    cp, cd = chans[args.photo], chans[args.drain]
    tp = np.asarray(cp.times); tp = tp - tp[0]
    vp = np.asarray(cp.volts)
    td = np.asarray(cd.times); td = td - td[0]
    vd = np.asarray(cd.volts)

    # 1. omega(t) from tape pulses / テープパルスからω(t)
    edges = tape_pulses(tp, vp)
    print(f"tape pulses: {len(edges)}")
    om = 2 * np.pi / np.diff(edges)
    tm = (edges[1:] + edges[:-1]) / 2

    # steady & cutoff / 定常とカット検出
    w_steady = np.median(om[om > 0.9 * om.max()])
    t_cut = tm[om > 0.98 * w_steady][-1]
    coast = tm > t_cut + 0.01
    print(f"steady ω = {w_steady:.0f} rad/s, cutoff at {t_cut:.3f} s, coast samples: {coast.sum()}")

    # 2. V_drain averaged per revolution / 回転周期ごとのドレイン電圧平均
    vbar = np.array([vd[(td >= a) & (td < b)].mean() for a, b in zip(edges[:-1], edges[1:])])

    # V_BAT from the post-stop DC level / 停止後DCレベル = V_BAT
    t_end = edges[-1]
    tail = td > t_end + 0.3
    vbat = vd[tail].mean() if tail.sum() > 100 else np.nan
    print(f"V_BAT (post-stop DC) = {vbat:.3f} V")

    # 3. line fit on coast portion / コースト区間の直線フィット
    x, y = om[coast], vbar[coast]
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ke = -slope * AMP_FIX   # RigolWFM 20→25 levels/div 補正 / amplitude fix
    yhat = A @ [slope, icpt]
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    print("\n== result ==")
    print(f"Ke = {ke:.4e} V·s/rad   (fit R² = {r2:.4f}, AMP_FIX=20/25 適用済み)")
    print(f"intercept = {icpt:.3f} V  (V_BAT実測 {vbat:.3f} V と一致するはず)")
    print(f"判定: 論文系 Ke=5.35e-4 / firmware系 6.125e-4 に対して → {ke:.3e}")
    R_PAPER = 0.593
    print(f"電気ブレーキ Ke²/R (R=0.593) = {ke*ke/R_PAPER:.3e} N·m·s/rad")
    JCQ = 335.3
    CQ = 4.10e-11
    W_H = 3670.0
    tau_eff = (1.375e-8) / (ke * ke / R_PAPER + 2 * CQ * W_H)
    print(f"→ τ_eff(ホバ, γ=1) = {tau_eff*1e3:.1f} ms")

    # 4. plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(tm, om, ".", ms=2); axes[0].axvline(t_cut, color="r", ls="--")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("ω [rad/s]"); axes[0].grid(alpha=0.3)
    axes[1].plot(x, y, ".", ms=3, label="coast data")
    xs = np.linspace(0, x.max(), 50)
    axes[1].plot(xs, icpt + slope * xs, "r-", lw=1.5, label=f"fit: Ke={ke:.3e}")
    if not np.isnan(vbat):
        axes[1].axhline(vbat, color="gray", ls=":", label=f"V_BAT={vbat:.2f}V")
    axes[1].set_xlabel("ω [rad/s]"); axes[1].set_ylabel("V_drain [V]")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = args.wfm.rsplit("/", 1)[-1].replace(".wfm", "") + "_ke_fit.png"
    fig.savefig(out, dpi=130)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    main()
