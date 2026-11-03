#!/usr/bin/env python3
"""
Xotiradan katta sinov natijasini analitik tekshirish.

Asl to'plam k marta ulanganda: kataklar soni m' = k m, gistogramma k marta oshadi
(shuning uchun persentil chegaralari L_j, U_j o'zgarmaydi), R'_j = k R_j, Q'_j = k Q_j.
Teorema 2.1 bo'yicha
    e'_j = (ln(k R_j) - Q_j / R_j) / ln(k m),
shuning uchun ulangan faylning EWM vaznlari asl fayldagi (R_j, Q_j, m) dan aniq bashorat
qilinadi. Skript bashoratni xotiradan katta fayl bo'yicha e1_passes.py natijasi
(summary.json -> env.w_ref) bilan solishtiradi.

Misol:
  python verify_replicated.py --src data\\orol_v2_30m\\data.npy --k 4 ^
      --dirs + - - + - + - - --summary experiments\\e3\\x4_30m\\summary.json
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_passes import blocks, bounds_from, f_hist, f_rq, f_stats, w_from_rq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--nbins", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    S = json.load(open(args.summary, encoding="utf-8"))
    env = S["env"]
    alpha = args.alpha if args.alpha is not None else env["alpha"]
    block = args.block if args.block is not None else env["block"]
    nodata = env["nodata"] if env["nodata"] is not None else np.nan
    w_big = np.array(env["w_ref"])

    X = np.load(args.src, mmap_mode="r")
    n, K = X.shape
    dirs = np.array([1 if d == "+" else -1 for d in args.dirs], np.int64)
    blk = blocks(K, block)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        res = [f.result() for f in [pool.submit(f_stats, X[:, lo:hi], nodata) for lo, hi in blk]]
        m = sum(r[0] for r in res)
        mn = np.min(np.stack([r[1] for r in res]), axis=0)
        mx = np.max(np.stack([r[2] for r in res]), axis=0)
        H = np.zeros((n, args.nbins), np.int64)
        for f in [pool.submit(f_hist, X[:, lo:hi], nodata, mn, mx, args.nbins) for lo, hi in blk]:
            H += f.result()
        L, U = bounds_from(H, mn, mx, m, alpha)
        R = np.zeros(n)
        Q = np.zeros(n)
        for f in [pool.submit(f_rq, X[:, lo:hi], nodata, L, U, dirs, args.chunk) for lo, hi in blk]:
            r, q = f.result()
            R += r
            Q += q

    w_orig = w_from_rq(R, Q, m, L, U)
    w_pred = w_from_rq(args.k * R, args.k * Q, args.k * m, L, U)
    d = np.abs(w_pred - w_big)
    print(f"asl fayl: m = {m}; ulangan fayl: m' = {args.k * m} (summary: K = {env['K']})")
    print(f"{'mezon':>14} {'asl w':>12} {'bashorat w':>12} {'o`lchangan w':>12} {'|farq|':>10}")
    names = list(env.get("criteria", {}).keys()) or [f"mezon_{j + 1}" for j in range(n)]
    for j in range(n):
        print(f"{names[j]:>14} {w_orig[j]:12.8f} {w_pred[j]:12.8f} {w_big[j]:12.8f} {d[j]:10.2e}")
    print(f"max |bashorat - o'lchangan| = {d.max():.3e}")
    out = dict(k=args.k, m=int(m), w_orig=w_orig.tolist(), w_pred=w_pred.tolist(), w_measured=w_big.tolist(),
               max_abs_diff=float(d.max()))
    path = os.path.join(os.path.dirname(os.path.abspath(args.summary)), "verify_replicated.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"yozildi: {path}")


if __name__ == "__main__":
    main()
