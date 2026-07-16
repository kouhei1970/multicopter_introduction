#!/usr/bin/env python3
"""Two-point per-capture calibration for DS1054Z .wfm amplitudes.
DS1054Z .wfm の振幅を「収録ごとの2点較正」で正しい電圧に換算するユーティリティ。

背景（2026-07-16 確定）:
  同じ V/div 表示でも、収録によって raw→電圧の対応が異なることが実測で確定
  （3Vテスト信号: 58レベル=3.00V ≈20レベル/目盛 vs EMF収録: 103レベル=4.03V
  ≈25.6レベル/目盛。垂直微調整は不使用と確認済み。隠れ変数はアクイジション
  モード等の疑い）。よって振幅を使う解析は、**その収録自身の中の既知2点**で
  較正するのが唯一確実な方法。時刻軸は常に正確。

使い方 / Usage:
    # 1) プレート自動検出モード: 波形中の2大プレートを検出し、
    #    それぞれの真の電圧（スコープのカーソル読み or 既知値）を渡す
    python3 wfm_two_point_cal.py capture.wfm --ch 2 --v-low 0.0 --v-high 3.0

    # 2) raw値指定モード: プレートでない任意の2点（rawと真値のペア）
    python3 wfm_two_point_cal.py capture.wfm --ch 2 --points 41:0.05 144:4.08

出力: 較正式 V = a·raw + b と、RigolWFM換算値からの補正式
      V_true = A·V_rigolwfm + B（既存スクリプトへの適用用）。
      --save で較正済み電圧を .npz に保存。
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np


def find_plateaus(raw):
    """detect the two dominant plateaus / 2大プレートの生値を検出"""
    hist = np.bincount(raw, minlength=256)
    # smooth ±1 and take two well-separated peaks
    peaks = []
    order = np.argsort(hist)[::-1]
    for lv in order:
        if hist[lv] < len(raw) * 0.01:
            break
        if all(abs(int(lv) - p) > 10 for p in peaks):
            peaks.append(int(lv))
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        raise SystemExit("プレートを2つ検出できませんでした。--points で指定してください")
    lo, hi = sorted(peaks)
    # refine by local weighted mean ±2 / 近傍加重平均で小数レベルに精密化
    def refine(c):
        w = hist[max(0, c - 2):c + 3].astype(float)
        x = np.arange(max(0, c - 2), c + 3)
        return float((w * x).sum() / w.sum())
    return refine(lo), refine(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wfm")
    ap.add_argument("--ch", type=int, default=1)
    ap.add_argument("--v-low", type=float, help="低プレートの真の電圧 [V]")
    ap.add_argument("--v-high", type=float, help="高プレートの真の電圧 [V]")
    ap.add_argument("--points", nargs=2, metavar="RAW:VOLT",
                    help="raw:true_volt を2点（プレート自動検出を使わない場合）")
    ap.add_argument("--save", action="store_true", help="較正済み電圧を .npz 保存")
    args = ap.parse_args()

    import RigolWFM.wfm as rigol
    w = rigol.Wfm.from_file(args.wfm, "1000Z")
    ch = {c.channel_number: c for c in w.channels}[args.ch]
    raw = np.asarray(ch.raw)
    t = np.asarray(ch.times)
    v_lib = np.asarray(ch.volts)

    if args.points:
        (r1, v1), (r2, v2) = [tuple(map(float, p.split(":"))) for p in args.points]
    else:
        if args.v_low is None or args.v_high is None:
            raise SystemExit("--v-low/--v-high か --points を指定してください")
        r1, r2 = find_plateaus(raw)
        v1, v2 = args.v_low, args.v_high
        print(f"検出プレート: raw_low={r1:.2f}, raw_high={r2:.2f}")

    a = (v2 - v1) / (r2 - r1)
    b = v1 - a * r1
    print(f"較正式:  V_true = {a:.6f} · raw + ({b:+.4f})")
    print(f"  等価表現: {ch.volt_per_division/a:.2f} レベル/目盛（V/div={ch.volt_per_division}）")

    # relation to RigolWFM volts: v_lib = a_lib·raw + b_lib
    idx = np.random.RandomState(0).choice(len(raw), min(20000, len(raw)), replace=False)
    A_ = np.vstack([raw[idx], np.ones(len(idx))]).T
    (a_lib, b_lib), *_ = np.linalg.lstsq(A_, v_lib[idx], rcond=None)
    A_corr = a / a_lib
    B_corr = b - A_corr * b_lib
    print(f"RigolWFM出力の補正式:  V_true = {A_corr:.5f} · V_rigolwfm + ({B_corr:+.4f})")

    if args.save:
        out = args.wfm.rsplit("/", 1)[-1].replace(".wfm", f"_ch{args.ch}_cal.npz")
        np.savez_compressed(out, t=t, volts=a * raw.astype(np.float64) + b)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
