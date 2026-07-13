#!/usr/bin/env python3
"""Roll divergence time of a StampFly with a tiny left/right prop speed mismatch.
左右プロペラ角速度のわずかなズレによる StampFly のロール発散時間の計算。

Question (from the teacher, 2026-07-14):
  ホバリング中の StampFly で、左右のプロペラ角速度が 0.1% だけズレていたら、
  水平から 90° に傾くまで何秒かかるか？

Model assumptions / モデルの仮定:
  - Hover: total thrust = m*g, each of 4 motors provides m*g/4
  - Thrust model: T = Ct * omega^2  (firmware motor_model.hpp と同じ)
  - Left pair spins at omega0*(1+delta), right pair at omega0 (delta = 0.1%)
  - Motors stay at those speeds (no control, no battery sag)
  - Rigid body, roll axis only: Ixx * phi_ddot = tau (constant torque)
  - Neglected: aerodynamic damping, translation coupling, gyroscopic effects
    (all small at the beginning of the divergence; damping would slightly
     lengthen the time, translation coupling shortens the recovery margin)

Parameter provenance / パラメータの出所 (すべて先生のリポジトリの一次情報):
  m    : 36.8 g   -- M5StamFly_spec_ja.md「製品重量」(firmware config.hpp は 0.037 kg)
  Ixx  : 9.16e-6  -- stampfly_ecosystem firmware/vehicle/tasks/api_task.cpp kSpecInertia[0]
                     (firmware/workshop lesson_06_modeling README の表と一致)
  d    : 0.023 m  -- vehicle_old sf_algo_control control_allocation.hpp QuadConfig
                     (X配置、モータ位置 x,y = ±0.023 m → ロールの横方向アームは 0.023 m)
  Ct   : 1.0e-8   -- vehicle_old sf_algo_control motor_model.cpp [N/(rad/s)^2]

Note: the final answer does NOT depend on Ct. Ct*omega0^2 = m*g/4 (hover条件)
なので、トルクは tau = m*g*delta_eff*d となり Ct と omega0 は打ち消し合う。
Ct は途中経過の omega0 (何 rad/s で回っているか) の表示にだけ使う。
"""

import math

# --- Parameters (see provenance above) ---
G = 9.81            # gravitational acceleration [m/s^2]
M = 0.0368          # mass [kg]
IXX = 9.16e-6       # roll moment of inertia [kg m^2]
D = 0.023           # lateral moment arm [m]
CT = 1.0e-8         # thrust coefficient [N/(rad/s)^2]
DELTA = 0.001       # left/right angular-velocity mismatch (0.1%)

# --- Hover condition ---
T_total = M * G                 # total thrust [N]
T0 = T_total / 4.0              # per-motor hover thrust [N]
omega0 = math.sqrt(T0 / CT)     # hover angular velocity [rad/s]

# --- Mismatch: left pair at omega0*(1+DELTA), right pair at omega0 ---
T_left = CT * (omega0 * (1.0 + DELTA)) ** 2   # per-motor, left [N]
T_right = CT * omega0**2                      # per-motor, right [N]
dT = T_left - T_right                         # per-motor thrust difference [N]

# Roll torque: 2 motors per side, arm D
tau = 2.0 * dT * D              # [N m]
alpha = tau / IXX               # angular acceleration [rad/s^2]

# Constant torque -> phi(t) = 0.5 * alpha * t^2
def t_to_angle(phi_deg: float) -> float:
    return math.sqrt(2.0 * math.radians(phi_deg) / alpha)

print(f"hover thrust total      : {T_total*1e3:8.2f} mN  ({M*1e3:.1f} gf相当)")
print(f"hover thrust per motor  : {T0*1e3:8.2f} mN")
print(f"hover omega0            : {omega0:8.1f} rad/s  ({omega0*60/(2*math.pi):,.0f} rpm)")
print(f"0.1% of omega0          : {omega0*DELTA:8.2f} rad/s ({omega0*DELTA*60/(2*math.pi):.0f} rpm)")
print(f"thrust diff per motor   : {dT*1e3:8.4f} mN  ({dT/G*1e6:.0f} mgf 相当)")
print(f"roll torque tau         : {tau:.3e} N m")
print(f"angular accel alpha     : {alpha:8.3f} rad/s^2")
print()
for phi in (1, 5, 10, 45, 90):
    print(f"time to {phi:3d} deg : {t_to_angle(phi):6.2f} s")
