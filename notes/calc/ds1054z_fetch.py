#!/usr/bin/env python3
"""DS1054Z LAN waveform fetcher (SCPI raw socket, no external deps).
DS1054Z から LAN 経由で波形を取得するツール（SCPI 生ソケット、追加ライブラリ不要）。

なぜこれか（2026-07-16）: .wfm ファイルは「画面座標系の収録メモリ」で、
raw→電圧の目盛り定数が非公開かつ収録構成依存（1ch≈20、2ch≈25レベル/目盛）と判明。
SCPI の :WAV:DATA? は文書化されたプリアンブル（YINCrement/YORigin/YREFerence）を
毎回スコープ自身が返すため、換算の曖昧さが原理的に消える。

Usage / 使い方:
    # 接続テスト（機種名が返ればOK）
    python3 ds1054z_fetch.py 192.168.1.xx --idn

    # 画面相当の波形（1200点、動作中でも可）を素早く取得
    python3 ds1054z_fetch.py <IP> --ch 1 2 --screen --out quick.npz

    # 深メモリ全点（要: 停止状態。実験後に STOP してから）
    python3 ds1054z_fetch.py <IP> --ch 1 2 --raw --out capture.npz

    # 補助: 取得前後の制御
    python3 ds1054z_fetch.py <IP> --stop        # 収録停止
    python3 ds1054z_fetch.py <IP> --run         # 収録再開
    python3 ds1054z_fetch.py <IP> --single      # シングルトリガ待機

    # 検証モード: 3Vテスト信号のプレートを読んで換算の正しさを自己確認
    python3 ds1054z_fetch.py <IP> --ch 1 --screen --verify3v

出力 .npz: t（秒）, ch<N>_v（電圧・スコープ公式換算）, 併せてプリアンブルも保存。
"""

import argparse
import socket
import struct
import sys
import time

import numpy as np


class DS1054Z:
    """Minimal SCPI-over-TCP client (port 5555). 最小限のSCPIクライアント"""

    def __init__(self, host, port=5555, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b""

    def cmd(self, s):
        """send command without reply / 応答なしコマンド"""
        self.sock.sendall((s + "\n").encode())

    def query(self, s):
        """send query, read one line / 1行応答のクエリ"""
        self.cmd(s)
        while b"\n" not in self.buf:
            self.buf += self.sock.recv(4096)
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode().strip()

    def read_block(self):
        """read a #N-length-prefixed binary block / TMCバイナリブロック読み"""
        while len(self.buf) < 2:
            self.buf += self.sock.recv(4096)
        assert self.buf[:1] == b"#", f"TMC block expected, got {self.buf[:16]!r}"
        ndig = int(self.buf[1:2])
        while len(self.buf) < 2 + ndig:
            self.buf += self.sock.recv(4096)
        nbytes = int(self.buf[2:2 + ndig])
        need = 2 + ndig + nbytes + 1          # +1 trailing \n
        while len(self.buf) < need:
            chunk = self.sock.recv(min(1 << 20, need - len(self.buf)))
            if not chunk:
                raise IOError("connection closed mid-block")
            self.buf += chunk
        data = self.buf[2 + ndig:2 + ndig + nbytes]
        self.buf = self.buf[need:]
        return data

    def preamble(self):
        """:WAV:PRE? → dict（文書化された換算パラメータ）"""
        f = self.query(":WAV:PRE?").split(",")
        keys = ["format", "type", "points", "count",
                "xincrement", "xorigin", "xreference",
                "yincrement", "yorigin", "yreference"]
        d = dict(zip(keys, f))
        for k in ("xincrement", "xorigin", "xreference",
                  "yincrement", "yorigin", "yreference"):
            d[k] = float(d[k])
        for k in ("format", "type", "points", "count"):
            d[k] = int(d[k])
        return d

    def fetch(self, ch, mode="RAW", chunk=250000):
        """fetch one channel's waveform as (t, volts, preamble).
        1チャネル分を（時刻, 電圧, プリアンブル）で取得。RAWは停止状態が必要。"""
        self.cmd(f":WAV:SOUR CHAN{ch}")
        self.cmd(f":WAV:MODE {mode}")
        self.cmd(":WAV:FORM BYTE")
        pre = self.preamble()
        n = pre["points"]
        raw = np.empty(n, dtype=np.uint8)
        got = 0
        while got < n:
            start, stop = got + 1, min(got + chunk, n)
            self.cmd(f":WAV:STAR {start}")
            self.cmd(f":WAV:STOP {stop}")
            self.cmd(":WAV:DATA?")
            block = self.read_block()
            raw[got:got + len(block)] = np.frombuffer(block, dtype=np.uint8)
            got += len(block)
        # ★ 文書化された公式換算（スコープ自身が返す係数。推測ゼロ）
        volts = (raw.astype(np.float64) - pre["yorigin"] - pre["yreference"]) * pre["yincrement"]
        t = pre["xorigin"] + np.arange(n) * pre["xincrement"]
        return t, volts, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host", help="オシロのIPアドレス（Utility→IO→LANで確認）")
    ap.add_argument("--ch", type=int, nargs="+", default=[1], help="チャネル番号（複数可）")
    ap.add_argument("--screen", action="store_true", help="画面相当1200点（動作中も可）")
    ap.add_argument("--raw", action="store_true", help="深メモリ全点（要・停止状態）")
    ap.add_argument("--out", default=None, help="保存先 .npz")
    ap.add_argument("--idn", action="store_true", help="接続テスト")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--verify3v", action="store_true",
                    help="プローブ補正信号(0/3V)でプレートを読み、換算を自己検証")
    args = ap.parse_args()

    sc = DS1054Z(args.host)
    print("接続:", sc.query("*IDN?"))
    if args.idn:
        return
    if args.stop:
        sc.cmd(":STOP"); print("STOP"); return
    if args.run:
        sc.cmd(":RUN"); print("RUN"); return
    if args.single:
        sc.cmd(":SING"); print("SINGLE armed"); return

    mode = "RAW" if args.raw else "NORM"
    if args.raw:
        status = sc.query(":TRIG:STAT?")
        if status != "STOP":
            print(f"! 深メモリ読みには停止が必要です（現在: {status}）。--stop してから再実行を")
            return

    out = {}
    for ch in args.ch:
        on = sc.query(f":CHAN{ch}:DISP?")
        if on.strip() not in ("1", "ON"):
            print(f"CH{ch}: 無効のためスキップ")
            continue
        t0 = time.time()
        t, v, pre = sc.fetch(ch, mode=mode)
        print(f"CH{ch}: {len(v):,}点  dt={pre['xincrement']:.3e}s  "
              f"YINC={pre['yincrement']:.6g} YOR={pre['yorigin']:.6g} YREF={pre['yreference']:.6g}  "
              f"V[{v.min():.3f},{v.max():.3f}]  ({time.time()-t0:.1f}s)")
        out["t"] = t
        out[f"ch{ch}_v"] = v
        for k, val in pre.items():
            out[f"ch{ch}_pre_{k}"] = val
        vdiv = float(sc.query(f":CHAN{ch}:SCAL?"))
        print(f"  V/div={vdiv} → YINC×レベル/目盛 = {vdiv/pre['yincrement']:.2f} レベル/目盛（参考）")

        if args.verify3v:
            hist, be = np.histogram(v, bins=200)
            order = np.argsort(hist)[::-1]
            plats = []
            for i in order:
                c = (be[i] + be[i + 1]) / 2
                if all(abs(c - p) > 0.5 for p in plats):
                    plats.append(c)
                if len(plats) == 2:
                    break
            lo, hi = sorted(plats)
            print(f"  [verify3v] プレート: {lo:.3f} V / {hi:.3f} V （期待 0.00/3.00）"
                  f" → 振幅 {hi-lo:.3f} V")

    if args.out and out:
        np.savez_compressed(args.out, **out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
