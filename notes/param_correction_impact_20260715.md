# モータ・プロペラ確定パラメータの影響調査（2026-07-15）

2026-07-15 の一連の実測（フォトインタラプタ回転計測、コーストダウン試験、プロペラ写真の
画素直接積分、回転子実測諸元）でモータ+プロペラのパラメータが確定した。
本レポートは stampfly_ecosystem 内でこの結果の影響を受ける箇所の棚卸しである。
**変更は未実施**（特定のみ）。経緯と再現スクリプトは
`multicopter_introduction/notes/qa_log.md` Q4-9〜Q4-13 および `notes/calc/` を参照。

## 1. 旧値 vs 新値（確定値）

| パラメータ | 旧値（コード/文書で流通） | **新値（実測確定）** | 比 新/旧 | 備考 |
|---|---|---|---|---|
| C_T [N/(rad/s)²] | 1.0e-8 | **6.7e-9** | ×0.67 | ベンチ推力×実測ω。旧値は旧プロペラ |
| C_Q [N·m·s²/rad²] | 9.71e-11 | **4.10e-11** | ×0.42 | コーストダウン J/C_Q=335 × 新J |
| κ = C_Q/C_T [m] | 9.71e-3 | **6.12e-3** | ×0.63 | 1/(4κ): 25.75 → 40.85 |
| J_mp（回転部合計）[kg·m²] | 2.01e-8（fw）/ 5.31e-8（論文） | **1.375e-8** | ×0.68 / ×0.26 | fwコメント「COMBINED」は結果的に正しい思想・値は32%過大 |
| J_prop | 2.01e-8 | **1.030e-8** | ×0.51 | 写真デジタイズ+画素積分+ピッチ0.9in補正 |
| J_rotor | 3.30e-8 | **3.45e-9** | ×0.10 | 旧値は桁違い（物理上限超え） |
| ω_hover [rad/s] | 2930〜3004 | **3670** | ×1.22〜1.25 | 35,000 rpm |
| モータ実効時定数（ホバ） | 0.02 s | **0.0175 s** | ≈維持 ✓ | **旧値が実測で裏付けられた** |
| ヨー零点 τ_z（ホバ） | 35.3 ms（yaw doc） | **45.7 ms** | ×1.29 | J/C_Q と ω_h のみに依存（頑健） |
| τ_z/τ_eff（リード比） | 3.0（yaw doc） | **2.60** | — | J 非依存 |
| クーロン摩擦 τ_c | （モデル化なし） | **9.5e-6 N·m** | 新規 | コーストダウン低速側で分離 |

補足: 電気系（R=0.593Ω, L_s=0.788µH, Ke=Kt=5.35e-4, τ_e=1.33µs）は論文同定のまま有効。
ただし genesis/vpython/yaw_axis_model.md は別系統の **Km=6.125e-4, Rm=0.34, Dm=3.69e-8** を
使用しており、論文値と並立している（§5 要調査）。

## 2. 影響箇所（カテゴリ別）

### A. 実機ファームウェア — 挙動に影響、変更は慎重に

| 箇所 | 旧値 | 影響 |
|---|---|---|
| `firmware/vehicle/components/sf_actuator/actuator.cpp:89` | KAPPA=0.00971 | **実機ミキシング**。κ×0.63 → 現行コードはヨートルクを1.59倍過大換算＝実ヨー出力は指令の63%。ただし飛行チューンされたヨーPIゲインが吸収している可能性が高く、**κだけ直すとヨー実効ゲインが1.59倍化**する。ゲイン再チューンとセットで |
| `firmware/vehicle_old/components/sf_algo_control/motor_model.cpp:13-14,19` `include/motor_model.hpp:49-50,61` | Ct=1.0e-8, Cq=9.71e-11, Jmp=2.01e-8 | thrustToOmega 等の変換が √(1.49)=1.22倍ズレ。閉ループで吸収されている可能性はあるが、モデルベース処理（フィードフォワード等）があれば直撃 |
| `firmware/vehicle_old/.../control_allocation.hpp:57` | kappa=9.71e-3 | 同上（旧機体のミキシング） |
| `firmware/vehicle_old/.../control_allocation.hpp:71` `main/config.hpp:529` `control_task.cpp:790` `vehicle/.../pid_controller.hpp:168-181,421` | max_thrust_per_motor=0.168N | **出所要確認**: ベンチ実測は duty0.9 で 11gf=0.108N/発。0.168N は過大の疑い（新C_Tだと ω=5,000rad/s 相当）。上限・T/W・ヨー authority 計算に波及 |

### B. シミュレータ — 物理定数の直接更新対象

| 箇所 | 旧値 |
|---|---|
| `simulator/sil/plant/plant.hpp:73,88` | Ct=1.00e-8, kappa=9.71e-3 |
| `simulator/sil/plant/plant.hpp:87` | **thrust_efficiency=1/1.12**: 旧C_Tの過大(×1.49)を吸収してきたファッジ係数の可能性大。新C_Tなら再導出（不要になるか値が変わる） |
| `simulator/genesis/motor_model.py:58,62` | Cq=9.71e-11, Jmp=2.01e-8 |
| `simulator/genesis/control_allocation.py:57,170-172` | kappa, thrust_to_duty(Ct=1.0e-8, Cq, Dm) |
| `simulator/vpython/core/motors.py:67,69` | Cq, Jmp |
| `simulator/vpython/scripts/run_sim.py:66-67` `run_vpython_headless.py:40-41,80` `run_sim_100hz_backup.py:69-70,117` | Ct/Cq/κ |

### C. 解析ツール — 換算値が系統的にズレる

| 箇所 | 旧値 |
|---|---|
| `tools/sysid/defaults.py:101,235-236` | J=2.01e-8, Cq, kappa |
| `tools/sysid/inertia.py:28-30` | CT/CQ/KAPPA |
| `tools/sysid/steady_state.py:105,287-288` | Cq デフォルト |
| `tools/log_analyzer/reconstruct_duties.py:43,48-49` | KAPPA/CT/CQ |
| `tools/log_analyzer/visualize_interactive.py:550,563-564,570` | kappa/Ct/Cq/max_thrust=0.168 |
| `analysis/scripts/acro_gain_conversion.py:10` `yaw_nt_kanazawa/torque_budget.py:62` | kappa |

### D. ドキュメント — 数値と例算の更新

| 箇所 | 内容 |
|---|---|
| `firmware/vehicle/docs/yaw_axis_model.md` | **内部不整合**: 冒頭(L12-14)は LHP=最小位相リード零点に訂正済み（フライトの+22〜32°リードとも整合）だが、本文§3〜§5に旧RHP記述が残存。パラメータも旧値（Jmp=2.01e-8, Cq=9.71e-11, ω0=2930 → T_m=11.8ms, τ_z=35.3ms, 比3.0）。新値では **τ_eff=17.5ms, τ_z=45.7ms, 比2.60**（結論の構造は不変）。要改稿 |
| `docs/architecture/control-system.md:883,1047,1097,1109,2252` | τ_m=0.02（維持✓、「推定値」→「実測確認済み」へ）、ω_m0=2930→3670、k_T=2C_tω_m0 の例算更新 |
| `docs/architecture/control-allocation-migration.md:115,129,162` | κ行列 0.00971、1/(4κ)=25.75→40.85、ω=√(0.0858/1.0e-8)=2930 の例算 |
| `firmware/workshop/lessons/lesson_06_modeling/README.md` | **更新済み**（2026-07-15、τ_m=0.02 復元+実測確認注記） |
| `firmware/workshop/lessons/lesson_07_sysid/README.md:76-78,176` | 参照解 τ_m=0.020 は維持✓（同定教材なので実測と整合していることがむしろ好都合） |
| `docs/workshop/slides/`（pptx/beamer/tikz） | tau_m=0.02 系は維持✓。Ct/Cq/J を印字している箇所は個別確認要 |

### E. 今回の実測が「裏付けた」もの — 変更不要（コメント格上げのみ）

- `simulator/sil/plant/plant.hpp:89` motor_tau=0.02 ✓（実測17.5ms）
- `tools/log_analyzer/rate_sysid.py:56` SPEC_MOTOR_T=0.02 ✓
- 各所の τ_m=0.02（lesson_07 参照解、docs、スライド）✓
- `firmware/vehicle_old/main/config.hpp:527` MASS=0.037 ✓（実測36.8g）
- 機体慣性 kSpecInertia=diag(9.16,13.3,20.4)e-6（2点吊り法）— 今回の対象外・有効

### F. 要調査（今回の測定では決着しない）

1. **電気系パラメータの並立**: genesis/vpython/yaw_axis_model.md の Km=6.125e-4, Rm=0.34,
   Dm=3.69e-8 vs 論文の Ke=Kt=5.35e-4, R=0.593, B=0。出所と測定条件の確認が必要
   （Rm=0.34 はドライバ込み/別個体/別手法？ Km 14%差・Rm 43%差は時定数計算に効く）。
2. **max_thrust_per_motor=0.168N の出所**（A参照）。
3. **ヨーPIゲインの依存関係**: κ修正を実機に入れる場合、`analysis/scripts/acro_gain_*` 系の
   ゲイン換算と`yaw_nt_kanazawa` の解析前提も連動確認。
4. 隣接リポジトリ: sf_sandbox / sf_motor の `data/current_parameters.md` は
   「旧プロペラ値」であることの明記が望ましい（論文本体は旧プロペラの実験として有効）。

## 3. 推奨する進め方

1. まず**シミュレータ（B）とツール（C）**を新値に更新（挙動リスクなし、精度向上のみ）。
   SIL の thrust_efficiency は新C_Tで再導出。
2. **ドキュメント（D）**の例算と yaw_axis_model.md の内部不整合を解消。
3. **実機（A）**は最後: κ・Ct の修正は飛行ゲインの再チューニング（または autotune 再実行）と
   セットで計画。max_thrust の出所確認を先に。
4. F の電気系並立は、次回のベンチ測定（V-I-ω を現行個体で1セット）で決着可能。
