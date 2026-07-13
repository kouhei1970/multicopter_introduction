#!/usr/bin/env python3
"""If all four motors got the SAME duty, how much would their speeds differ?
「4モータに同じ Duty を入れたら回転数は何%ズレるか」をホバリングログから推定する。

Logic / 論理:
  1. ホバリング中、姿勢制御はロール・ピッチ・ヨーのトルク釣り合いを保っている。
     機体が水平で CG が中心にあるなら、各モータの推力はほぼ等しい（≈ mg/4）。
     推力が等しい ≈ 回転数 ω もほぼ等しい（プロペラの Ct が同一と仮定）。
  2. それなのに指令 Duty はモータごとに違う → 同じ ω を出すのに必要な Duty が
     個体ごとに違う（モータ+プロペラ+駆動回路を合わせた実効ゲインの個体差）。
  3. ベンチ実測の Duty–推力特性（sf_motor/data/measured_motor_spec.md,
     2026-03-17, V=3.7V, プロペラ付き）から局所感度 dlnω/dlnDuty を求め、
     「同じ Duty を与えたときの回転数差」に換算する。
     測定では T ∝ Duty^~1.0、モデル T = Ct·ω² より ω ∝ Duty^~0.5。

Caveats / 注意（未確認事項は qa_log.md にも記載）:
  - ホバ Duty 差には CG オフセット由来のロール/ピッチトリム分も混入する。
    CW/CCW 対角パターン（ヨートリム）は CG では説明できず、個体差・系統差由来。
  - ベンチ特性は 1 個のモータの実測。感度（傾き）だけを借りる。
  - 飛行中の電池電圧はベンチの 3.7V と異なるが、相対感度への影響は小さいと仮定。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hover_duty_stats import analyze  # noqa: E402

# --- Bench duty->thrust table (measured_motor_spec.md, 2026-03-17) ---
BENCH = {0.1: 1.00, 0.2: 2.00, 0.3: 3.25, 0.4: 4.50, 0.5: 5.75,
         0.6: 7.00, 0.7: 8.00, 0.8: 9.25, 0.9: 11.00}  # duty -> thrust [gf]

# Local log-log slope of thrust vs duty around hover duty (~0.7)
# ホバ付近 (0.6-0.8) の対数傾き: T ∝ duty^n
N_T = (math.log(BENCH[0.8]) - math.log(BENCH[0.6])) / (math.log(0.8) - math.log(0.6))
N_OMEGA = 0.5 * N_T  # T = Ct·ω² -> ω ∝ duty^(n/2)

# --- StampFly rigid-body parameters (provenance: see roll_divergence.py) ---
G, M, D = 9.81, 0.0368, 0.023
IXX, IYY = 9.16e-6, 13.3e-6
T0 = M * G / 4.0  # per-motor hover thrust [N]

# Motor geometry (control_allocation.hpp): index 0..3 = M1:FR, M2:RR, M3:RL, M4:FL
RIGHT, LEFT = (0, 1), (2, 3)   # roll pairs
FRONT, REAR = (0, 3), (1, 2)   # pitch pairs


def flip_time(torque: float, inertia: float) -> float:
    """Time for phi = 90 deg under constant torque. 一定トルクで90°に達する時間."""
    return math.sqrt(math.pi * inertia / torque) if torque > 0 else float("inf")


print(f"bench slope: T ∝ duty^{N_T:.2f}  ->  ω ∝ duty^{N_OMEGA:.2f}")

for path in sys.argv[1:]:
    means = analyze(path)
    if means is None:
        continue
    d_mean = sum(means) / 4.0
    # Speed deviation of each motor if given the common duty d_mean.
    # 各モータに共通 Duty を与えたときの回転数偏差。
    # 弱いモータ（ホバで多く Duty が要る）は同一 Duty では遅く回る → 符号は負。
    dev = [-N_OMEGA * (m - d_mean) / d_mean for m in means]
    print("  same-duty speed deviation / 同一Dutyでの回転数偏差:")
    for name, s in zip(["M1:FR", "M2:RR", "M3:RL", "M4:FL"], dev):
        print(f"    {name}  {100*s:+.2f}%")
    spread = max(dev) - min(dev)
    print(f"    max-min spread: {100*spread:.2f}%")

    # Thrust deviation ≈ 2×speed deviation (T ∝ ω²) -> axis torques
    thrust = [T0 * (1.0 + s) ** 2 for s in dev]
    tau_roll = abs(sum(thrust[i] for i in LEFT) - sum(thrust[i] for i in RIGHT)) * D
    tau_pitch = abs(sum(thrust[i] for i in FRONT) - sum(thrust[i] for i in REAR)) * D
    print(f"    roll  torque {tau_roll:.2e} N·m -> 90° in {flip_time(tau_roll, IXX):.2f} s")
    print(f"    pitch torque {tau_pitch:.2e} N·m -> 90° in {flip_time(tau_pitch, IYY):.2f} s")
