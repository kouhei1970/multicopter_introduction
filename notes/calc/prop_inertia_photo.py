#!/usr/bin/env python3
"""Propeller moment of inertia from the top-view photo (teacher's method).
プロペラ上面写真からの回転軸周り慣性モーメント推定（先生指定の方法）。

Inputs / 入力:
  assets/IMG_3579.JPG — 現行4枚プロペラの上面写真（白背景、定規同一平面）
  実測質量（先生提供 2026-07-15）: ハブ 0.14 g、ブレード1枚 0.02375 g

Method / 方法（先生の指示）:
  1. 写真から外形をデジタイズ（色分離: ハブ=不透明濃赤 G<60、ブレード=半透明ピンク）
  2. スケールは定規の1mm目盛り間隔（自己相関で推定）
  3. ブレードを半径方向に微小スライスし、各区分を台形近似
     各区分の慣性 = 質量×(重心半径² + 自身中心周り項)、を合計
  4. 検算としてピクセル直接積分（スライス幅→0の極限）も併記

Assumptions / 仮定（結果に系統誤差として乗り得る）:
  - 面密度一様（材料厚一定）。ピッチ（ねじり）による投影の縮みは無視
    → 投影面積あたり質量は根元ほど実際は大きい可能性（後述の感度参照）
  - ハブは一様断面の円柱（投影に一様面密度）。シャフト穴は無視（小径）
  - カメラは十分遠方・垂直（透視歪み小）。スケール誤差±3%なら I は±6%
"""

import numpy as np
from PIL import Image
from scipy import ndimage

IMG = "/Users/kouhei/tmp/github/multicopter_introduction/assets/IMG_3579.JPG"
M_HUB = 0.14e-3          # [kg]
M_BLADE = 0.02375e-3     # [kg] per blade
N_BLADES = 4

a = np.asarray(Image.open(IMG)).astype(float)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
redness = R - (G + B) / 2

# ---------- 1. scale from ruler ticks / 定規からスケール ----------
strip = a[:, 3140:3260, :].mean(axis=(1, 2))
s = strip - np.convolve(strip, np.ones(201) / 201, mode="same")
s = s[300:2800]
ac = np.correlate(s, s, mode="full")[len(s) - 1:]
lag0 = 20 + np.argmax(ac[20:200])
# parabolic sub-pixel refinement / 放物線補間でサブピクセル化
y0, y1, y2 = ac[lag0 - 1], ac[lag0], ac[lag0 + 1]
lag = lag0 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
PX_PER_MM = lag
print(f"scale: {PX_PER_MM:.2f} px/mm (1mm tick autocorrelation)")

# ---------- 2. segmentation / 色分離 ----------
prop = redness > 25
# largest connected component / 最大連結成分のみ
lab, n = ndimage.label(prop)
sizes = ndimage.sum(prop, lab, range(1, n + 1))
prop = lab == (1 + np.argmax(sizes))
hub = prop & (G < 60)
# fill hub holes (specular highlights) / ハブ内の白飛びを埋める
hub = ndimage.binary_fill_holes(ndimage.binary_closing(hub, iterations=5))
blade = prop & ~hub

ys, xs = np.nonzero(hub)
cx, cy = xs.mean(), ys.mean()
A_hub_mm2 = hub.sum() / PX_PER_MM**2
A_blade_mm2 = blade.sum() / PX_PER_MM**2
R_hub_mm = np.sqrt(A_hub_mm2 / np.pi)
print(f"hub: area {A_hub_mm2:.1f} mm² (equiv radius {R_hub_mm:.2f} mm), centroid ({cx:.0f},{cy:.0f})")
print(f"blades: total area {A_blade_mm2:.1f} mm² ({A_blade_mm2/N_BLADES:.1f} mm²/blade)")

# radius map in mm / 半径マップ
Y, X = np.nonzero(prop)
r_all = np.hypot(X - cx, Y - cy) / PX_PER_MM
r_blade = np.hypot(np.nonzero(blade)[1] - cx, np.nonzero(blade)[0] - cy) / PX_PER_MM
r_hub = np.hypot(np.nonzero(hub)[1] - cx, np.nonzero(hub)[0] - cy) / PX_PER_MM
R_tip = np.percentile(r_blade, 99.95)
print(f"blade tip radius: {R_tip:.2f} mm (prop diameter {2*R_tip:.1f} mm)")

# ---------- 3. teacher's method: radial slices + trapezoids ----------
# 半径方向スライス＋台形近似（4枚まとめた合計コード幅 w(r) を使う）
sigma_b = (N_BLADES * M_BLADE) / (A_blade_mm2 * 1e-6)     # [kg/m²]
sigma_h = M_HUB / (A_hub_mm2 * 1e-6)
print(f"areal density: blade {sigma_b:.3f} kg/m², hub {sigma_h:.3f} kg/m²")

def slice_inertia(dr_mm):
    """trapezoid-slice integration / 台形スライス積分"""
    edges = np.arange(0, R_tip + dr_mm, dr_mm)
    # total chord width w(r) at slice edges, from pixel counts in thin annuli
    # 薄い環帯のピクセル数から各エッジ半径での合計コード幅を推定
    w_edge = []
    thin = dr_mm / 4
    for re in edges:
        m = (r_blade >= re - thin / 2) & (r_blade < re + thin / 2)
        w_edge.append(m.sum() / PX_PER_MM**2 / thin)   # [mm]
    w_edge = np.array(w_edge)
    I = 0.0
    rows = []
    for k in range(len(edges) - 1):
        r1, r2 = edges[k], edges[k + 1]
        w1, w2 = w_edge[k], w_edge[k + 1]
        A = 0.5 * (w1 + w2) * dr_mm                     # trapezoid area [mm²]
        if A <= 0:
            continue
        m_i = sigma_b * A * 1e-6                        # [kg]
        # trapezoid centroid radius / 台形の重心半径
        rc = r1 + dr_mm * (w1 + 2 * w2) / (3 * (w1 + w2))
        # own-center inertia: radial extent only. NOTE: annular slices have no
        # chordwise term — every point of the annulus is at distance ~r from
        # the axis (the "width" lies along the arc).
        # 自身中心周り: 半径方向 Δr²/12 のみ。環帯スライスではコード方向の
        # 広がりは軸距離を変えない（幅は円弧に沿うため）ので w²/12 項は不要。
        I_own = m_i * (dr_mm * 1e-3) ** 2 / 12.0
        I += m_i * (rc * 1e-3) ** 2 + I_own
        rows.append((r1, r2, w1, w2, m_i, rc))
    return I, rows

I_b_slice, rows = slice_inertia(1.0)
print("\n== teacher's method: 1mm trapezoid slices (blades) ==")
print("  r1-r2 [mm]   w1→w2 [mm]   m [mg]   rc [mm]")
for r1, r2, w1, w2, m_i, rc in rows:
    print(f"  {r1:4.1f}-{r2:4.1f}   {w1:5.2f}→{w2:5.2f}   {m_i*1e6:6.3f}   {rc:5.2f}")
for dr in (2.0, 1.0, 0.5, 0.25):
    I_d, _ = slice_inertia(dr)
    print(f"  Δr = {dr:4.2f} mm: I_blades = {I_d:.3e} kg·m²")

# ---------- 4. pixel-exact integration / ピクセル直接積分（極限検算） ----------
apx = (1e-3 / PX_PER_MM) ** 2                            # pixel area [m²]
I_b_px = sigma_b * apx * np.sum((r_blade * 1e-3) ** 2)
I_h_px = sigma_h * apx * np.sum((r_hub * 1e-3) ** 2)
print(f"\npixel-exact: I_blades = {I_b_px:.3e},  I_hub = {I_h_px:.3e} kg·m²")

I_total = I_b_px + I_h_px
print("\n========== RESULT ==========")
print(f"I_hub    = {I_h_px:.3e} kg·m²  (mass {M_HUB*1e3:.2f} g, R_eq {R_hub_mm:.2f} mm)")
print(f"I_blades = {I_b_px:.3e} kg·m²  (mass {N_BLADES*M_BLADE*1e3:.3f} g)")
print(f"I_prop   = {I_total:.3e} kg·m²")
print(f"radius of gyration (blades): {np.sqrt(I_b_px/(N_BLADES*M_BLADE))*1e3:.2f} mm")
print(f"（参考）旧推定 J_prop = 2.01e-8 との比: {I_total/2.01e-8:.2f}")

# ---------- 5. diagnostics / 診断画像 ----------
ov = a.copy().astype(np.uint8)
ov[blade] = [0, 160, 255]
ov[hub] = [255, 220, 0]
Image.fromarray(ov).resize((1008, 756)).save(
    "/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/prop_seg_overlay.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
hist, be = np.histogram(r_blade, bins=int(R_tip / 0.25))
w_of_r = hist / PX_PER_MM**2 / 0.25
ax.step(be[:-1], w_of_r, where="post", label="total chord width w(r) (4 blades)")
ax.set_xlabel("radius r [mm]"); ax.set_ylabel("w(r) [mm]")
ax.set_title("blade planform digitized from photo")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("/Users/kouhei/tmp/github/multicopter_introduction/notes/calc/prop_planform.png", dpi=130)
print("diagnostics saved: prop_seg_overlay.png, prop_planform.png")
