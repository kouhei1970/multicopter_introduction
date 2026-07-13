#!/usr/bin/env python3
"""Per-motor hover duty statistics from StampFly UDP telemetry logs.
StampFly の UDP テレメトリログからホバリング中のモータ別 Duty 統計を出す。

Data: stampfly_ecosystem/logs/stampfly_udp_*.jsonl
  - id=="ctrl_ref" records carry motor_duty:[M1,M2,M3,M4]
  - Motor layout (control_allocation.hpp): M1:FR, M2:RR, M3:RL, M4:FL
    motor_dir = {-1,1,-1,1} -> M1:CCW, M2:CW, M3:CCW, M4:CW

Hover extraction: keep samples where all four duties are in (0.2, 0.95),
then trim the first/last 20% of that flight segment to avoid takeoff/landing
transients. No other filtering — the per-motor MEAN difference is the signal.
ホバ区間抽出: 4モータとも duty が (0.2,0.95) のサンプルを飛行区間とし、
離着陸の過渡を避けるため前後20%を捨てる。それ以外の加工はしない。
"""

import json
import sys
import statistics

MOTOR_NAMES = ["M1:FR(CCW)", "M2:RR(CW)", "M3:RL(CCW)", "M4:FL(CW)"]


def analyze(path: str):
    duties = []
    with open(path) as f:
        for line in f:
            if '"ctrl_ref"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = rec.get("motor_duty")
            if d and all(0.2 < x < 0.95 for x in d):
                duties.append(d)
    if len(duties) < 100:
        print(f"{path}: hover samples too few ({len(duties)}) -- skip")
        return None

    n = len(duties)
    core = duties[n // 5 : -n // 5]  # trim 20% both ends
    means = [statistics.mean(col) for col in zip(*core)]
    sds = [statistics.stdev(col) for col in zip(*core)]
    grand = statistics.mean(means)

    print(f"\n{path}")
    print(f"  hover samples: {len(core)} (of {n} in-flight ctrl_ref records)")
    for name, m, s in zip(MOTOR_NAMES, means, sds):
        print(f"  {name:12s} mean duty = {m:.4f}  (sd {s:.4f})  dev from grand mean: {100*(m/grand-1):+.2f}%")
    ccw = (means[0] + means[2]) / 2
    cw = (means[1] + means[3]) / 2
    print(f"  CCW pair (M1,M3) mean: {ccw:.4f}")
    print(f"  CW  pair (M2,M4) mean: {cw:.4f}")
    print(f"  CW - CCW: {cw-ccw:+.4f}  ({100*(cw/ccw-1):+.2f}% relative)")
    print(f"  max-min across motors: {max(means)-min(means):.4f} duty ({100*(max(means)/min(means)-1):.2f}% relative)")
    return means


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
