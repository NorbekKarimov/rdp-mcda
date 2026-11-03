#!/usr/bin/env python3
"""
E2 — Determinizm va qaror barqarorligi tajribasi (1-maqola, IJGI protokoli).

Taqqoslanadi:
  P  : deterministik reduksiya (float64 va float32 to'plagichlar) — har bir blok natijasi Pi[k] ga yoziladi,
       birlashtirish bitta oqimda k = 0..P-1 tartibida (Algoritm 3.4).
  A3 : oddiy reduksiya — blok natijalari ishchilar TUGASH tartibida
       umumiy yig'indiga qo'shiladi (as_completed).
Ikkalasida ham bir xil Numba yadrolari ishlatiladi; farq faqat birlashtirishda.

Kirish: (a) .npy fayl (--data) — shakli (n, K), float32, C-tartib; np.load(mmap_mode="r")
       bilan o'qiladi, sarlavha hajmi va shakl fayldan avtomatik olinadi; yoki
       (b) sarlavhasiz binar fayl (--ard, --n, --K, ixtiyoriy --offset).
Nodata: --nodata berilmasa, fayl yonidagi meta.json dan ("nodata", "nodata_value",
       "no_data" kalitlari) olinadi; topilmasa faqat NaN yaroqsiz hisoblanadi.
Chiqish: runs.csv (har ishga tushirish bir qator) va summary.json.

Misol:
  python e2_determinism.py --data D:\\Norbek\\kmqq-majmua\\data\\orol_v2_30m\\data.npy \
      --dirs + - + + - + - - --threads 4 6 12 --runs 100 --out e2_d1_30

Yo'nalishlar (--dirs) loyiha konfiguratsiyasidagi bilan bir xil bo'lishi SHART.
"""
import argparse, csv, hashlib, json, os, platform, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import numba
from numba import njit

# ----------------------------------------------------------------- yadrolar
# fastmath=False (standart): Teorema 3.2 sharti.

@njit(nogil=True, cache=True)
def k_stats(X, nodata):
    n, b = X.shape
    cnt = 0
    mn = np.full(n, np.inf)
    mx = np.full(n, -np.inf)
    for i in range(b):
        ok = True
        for j in range(n):
            v = X[j, i]
            if not np.isfinite(v) or v == nodata:
                ok = False
                break
        if not ok:
            continue
        cnt += 1
        for j in range(n):
            v = np.float64(X[j, i])
            if v < mn[j]:
                mn[j] = v
            if v > mx[j]:
                mx[j] = v
    return cnt, mn, mx


@njit(nogil=True, cache=True)
def k_hist(X, nodata, mn, mx, nb):
    n, b = X.shape
    H = np.zeros((n, nb), np.int64)
    for i in range(b):
        ok = True
        for j in range(n):
            v = X[j, i]
            if not np.isfinite(v) or v == nodata:
                ok = False
                break
        if not ok:
            continue
        for j in range(n):
            w = mx[j] - mn[j]
            if w <= 0.0:
                h = 0
            else:
                h = int((np.float64(X[j, i]) - mn[j]) / w * nb)
                if h < 0:
                    h = 0
                elif h > nb - 1:
                    h = nb - 1
            H[j, h] += 1
    return H


@njit(nogil=True, cache=True)
def _norm(x, L, U, d):
    if U <= L:
        return 0.0
    if d > 0:
        r = (x - L) / (U - L)
    else:
        r = (U - x) / (U - L)
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0
    return r


@njit(nogil=True, cache=True)
def k_ewm64(X, nodata, L, U, dirs, c):
    """R_j, Q_j: float64, uch pog'onali (bo'lak ichida, bo'laklar bo'yicha)."""
    n, b = X.shape
    R = np.zeros(n)
    Q = np.zeros(n)
    rho = np.zeros(n)
    th = np.zeros(n)
    s = 0
    while s < b:
        e = min(s + c, b)
        rho[:] = 0.0
        th[:] = 0.0
        for i in range(s, e):
            ok = True
            for j in range(n):
                v = X[j, i]
                if not np.isfinite(v) or v == nodata:
                    ok = False
                    break
            if not ok:
                continue
            for j in range(n):
                r = _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
                rho[j] += r
                if r > 0.0:
                    th[j] += r * np.log(r)
        for j in range(n):
            R[j] += rho[j]
            Q[j] += th[j]
        s = e
    return R, Q


@njit(nogil=True, cache=True)
def k_ewm32(X, nodata, L, U, dirs, c):
    """Xuddi shu, lekin barcha to'plagichlar float32 (GPU float32 holatiga o'xshash)."""
    n, b = X.shape
    R = np.zeros(n, np.float32)
    Q = np.zeros(n, np.float32)
    rho = np.zeros(n, np.float32)
    th = np.zeros(n, np.float32)
    s = 0
    while s < b:
        e = min(s + c, b)
        rho[:] = 0.0
        th[:] = 0.0
        for i in range(s, e):
            ok = True
            for j in range(n):
                v = X[j, i]
                if not np.isfinite(v) or v == nodata:
                    ok = False
                    break
            if not ok:
                continue
            for j in range(n):
                r = np.float32(_norm(np.float64(X[j, i]), L[j], U[j], dirs[j]))
                rho[j] += r
                if r > 0.0:
                    th[j] += r * np.float32(np.log(r))
        for j in range(n):
            R[j] += rho[j]
            Q[j] += th[j]
        s = e
    return R, Q


@njit(nogil=True, cache=True)
def k_saw(X, nodata, L, U, dirs, w, out):
    n, b = X.shape
    for i in range(b):
        ok = True
        for j in range(n):
            v = X[j, i]
            if not np.isfinite(v) or v == nodata:
                ok = False
                break
        if not ok:
            out[i] = np.nan
            continue
        s = 0.0
        for j in range(n):
            s += w[j] * _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
        out[i] = np.float32(s)

# ------------------------------------------------------------ yordamchilar

def blocks(K, b):
    return [(lo, min(lo + b, K)) for lo in range(0, K, b)]


def hist_quantile(Hj, mn, mx, m, q):
    """Algoritm 3.2: gistogramma bo'yicha kvantil (chiziqli interpolyatsiya)."""
    nb = Hj.shape[0]
    if mx <= mn:
        return mn
    C = np.cumsum(Hj)
    t = q * m
    h = int(np.searchsorted(C, t, side="left"))
    h = min(h, nb - 1)
    prev = C[h - 1] if h > 0 else 0
    frac = (t - prev) / Hj[h] if Hj[h] > 0 else 0.0
    return mn + (h + frac) * (mx - mn) / nb


def weights_from_sums(R, Q, m, L, U):
    R = np.asarray(R, np.float64)
    Q = np.asarray(Q, np.float64)
    n = R.shape[0]
    d = np.zeros(n)
    for j in range(n):
        if U[j] <= L[j] or R[j] <= 0:
            d[j] = 0.0
        else:
            e = (np.log(R[j]) - Q[j] / R[j]) / np.log(m)
            d[j] = 1.0 - e
    if d.sum() == 0:
        return np.full(n, 1.0 / n)
    return d / d.sum()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

# -------------------------------------------------------------- o'tishlar

def pass_stats(X, blk, nodata, pool):
    futs = [pool.submit(k_stats, X[:, lo:hi], nodata) for lo, hi in blk]
    res = [f.result() for f in futs]  # min/max/butun son: tartibga bog'liq emas
    m = sum(r[0] for r in res)
    mn = np.min(np.stack([r[1] for r in res]), axis=0)
    mx = np.max(np.stack([r[2] for r in res]), axis=0)
    return m, mn, mx


def pass_hist(X, blk, nodata, mn, mx, nb, pool):
    futs = [pool.submit(k_hist, X[:, lo:hi], nodata, mn, mx, nb) for lo, hi in blk]
    H = np.zeros((X.shape[0], nb), np.int64)
    for f in futs:  # butun son: aniq va tartibga bog'liq emas (Lemma 3.2)
        H += f.result()
    return H


def pass_ewm(X, blk, nodata, L, U, dirs, c, pool, mode, acc):
    kern = k_ewm64 if acc == "f64" else k_ewm32
    dt = np.float64 if acc == "f64" else np.float32
    n = X.shape[0]
    futs = {pool.submit(kern, X[:, lo:hi], nodata, L, U, dirs, c): k
            for k, (lo, hi) in enumerate(blk)}
    if mode == "det":
        Pi_R = np.zeros((len(blk), n), dt)
        Pi_Q = np.zeros((len(blk), n), dt)
        for f in as_completed(futs):
            k = futs[f]
            Pi_R[k], Pi_Q[k] = f.result()
        R = np.zeros(n, dt)
        Q = np.zeros(n, dt)
        for k in range(len(blk)):  # qat'iy tartib sigma = (1..P)
            R += Pi_R[k]
            Q += Pi_Q[k]
    else:  # "naive": tugash tartibida qo'shish
        R = np.zeros(n, dt)
        Q = np.zeros(n, dt)
        for f in as_completed(futs):
            r, q = f.result()
            R += r
            Q += q
    return R, Q


def pass_saw(X, blk, nodata, L, U, dirs, w, pool, out):
    futs = [pool.submit(k_saw, X[:, lo:hi], nodata, L, U, dirs, w, out[lo:hi])
            for lo, hi in blk]
    for f in futs:
        f.result()

# ------------------------------------------------------------- bitta ishga tushirish

def run_chain(X, args, T, mode, acc, b, bounds):
    blk = blocks(X.shape[1], b)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=T) as pool:
        L, U, m = bounds
        R, Q = pass_ewm(X, blk, args.nodata, L, U, args.dirs_arr, args.chunk, pool, mode, acc)
        w = weights_from_sums(R, Q, m, L, U)
        S = np.empty(X.shape[1], np.float32)
        pass_saw(X, blk, args.nodata, L, U, args.dirs_arr, w, pool, S)
    return w, S, time.perf_counter() - t0


def topk_masks(S, fracs):
    v = S[np.isfinite(S)]
    out = {}
    for f in fracs:
        k = max(1, int(round(f / 100.0 * v.size)))
        thr = np.partition(v, v.size - k)[v.size - k]
        out[f] = S >= thr  # tenglar ham kiradi; NaN >= thr -> False
    return out


def jaccard(a, b):
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def classes(S, breaks):
    c = np.digitize(S, breaks).astype(np.int8)
    c[~np.isfinite(S)] = -1
    return c

# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help=".npy fayl, shakli (n, K)")
    src.add_argument("--ard", help="sarlavhasiz binar float32 fayl, shakli (n, K)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="--ard uchun sarlavha hajmi, bayt")
    ap.add_argument("--dirs", nargs="+", required=True, help="har mezon uchun + (benefit) yoki - (cost)")
    ap.add_argument("--nodata", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--bounds", default=None, help="JSON: {\"L\": [...], \"U\": [...]} — etalon chegaralar")
    ap.add_argument("--block", type=int, default=2 ** 20)
    ap.add_argument("--blocks_alt", type=int, nargs="*", default=[2 ** 18, 2 ** 22])
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--nbins", type=int, default=65536)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 6, 12])
    ap.add_argument("--threads_det", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--topk", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--breaks", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--out", default="e2_out")
    args = ap.parse_args()

    # ---- ma'lumotni ochish
    if args.data:
        X = np.load(args.data, mmap_mode="r")
        path = args.data
        if X.ndim != 2:
            sys.exit(f"Kutilgan shakl (n, K), fayldagi shakl {X.shape}")
        if args.n is not None and args.n != X.shape[0]:
            sys.exit(f"--n={args.n}, fayldagi n={X.shape[0]}")
        if args.K is not None and args.K != X.shape[1]:
            sys.exit(f"--K={args.K}, fayldagi K={X.shape[1]}")
        args.n, args.K = X.shape
        header = X.offset if hasattr(X, "offset") else None
    else:
        if args.n is None or args.K is None:
            sys.exit("--ard uchun --n va --K majburiy")
        X = np.memmap(args.ard, dtype=np.float32, mode="r", offset=args.offset,
                      shape=(args.n, args.K))
        path = args.ard
        header = args.offset
    if X.dtype != np.float32:
        print(f"OGOHLANTIRISH: dtype={X.dtype}, float32 kutilgan", flush=True)

    # ---- nodata
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
    nodata_src = "--nodata"
    if args.nodata is None:
        args.nodata = np.nan
        nodata_src = "yo'q (faqat NaN)"
        if isinstance(meta, dict):
            for key in ("nodata", "nodata_value", "no_data"):
                if meta.get(key) is not None:
                    args.nodata = float(meta[key])
                    nodata_src = f"meta.json:{key}"
                    break
    print(f"fayl: {path}\n  shakl (n, K) = {X.shape}, dtype = {X.dtype}, sarlavha = {header} bayt"
          f"\n  nodata = {args.nodata} ({nodata_src})"
          f"\n  meta.json kalitlari: {list(meta.keys()) if isinstance(meta, dict) else 'topilmadi'}",
          flush=True)

    if len(args.dirs) != args.n:
        sys.exit(f"--dirs da {len(args.dirs)} ta belgi, mezonlar soni n = {args.n}")
    args.dirs_arr = np.array([1 if d == "+" else -1 for d in args.dirs], np.int64)
    names = meta.get("names") if isinstance(meta, dict) else None
    if not names or len(names) != args.n:
        names = [f"mezon_{j + 1}" for j in range(args.n)]
    print("  mezonlar va yo'nalishlar:", ", ".join(f"{nm} ({d})" for nm, d in zip(names, args.dirs)),
          flush=True)
    os.makedirs(args.out, exist_ok=True)

    env = dict(python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
               platform=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
               data=os.path.abspath(path), header_bytes=header, n=args.n, K=args.K,
               dtype=str(X.dtype), nodata=None if np.isnan(args.nodata) else args.nodata,
               nodata_source=nodata_src, block=args.block, chunk=args.chunk,
               criteria=dict(zip(names, args.dirs)))

    # 1-2 o'tishlar: deterministik (butun son, min/max), bir marta hisoblanadi
    blk = blocks(args.K, args.block)
    with ThreadPoolExecutor(max_workers=max(args.threads)) as pool:
        m, mn, mx = pass_stats(X, blk, args.nodata, pool)
        if args.bounds:
            bd = json.load(open(args.bounds))
            L, U = np.array(bd["L"], np.float64), np.array(bd["U"], np.float64)
        else:
            H = pass_hist(X, blk, args.nodata, mn, mx, args.nbins, pool)
            L = np.array([hist_quantile(H[j], mn[j], mx[j], m, args.alpha) for j in range(args.n)])
            U = np.array([hist_quantile(H[j], mn[j], mx[j], m, 1 - args.alpha) for j in range(args.n)])
    bounds = (L, U, m)
    env.update(m_valid=int(m), L=L.tolist(), U=U.tolist())
    print(f"yaroqli kataklar m = {m}; bloklar P = {len(blk)}", flush=True)

    # Etalonlar: deterministik, T = 1, har bir aniqlik uchun alohida.
    # A3 ning tartib ta'siri o'z aniqligidagi etalon bilan solishtiriladi;
    # float32 ning float64 dan farqi (aniqlik ta'siri) alohida ustunda.
    print("etalonlar: det/T=1, f64 va f32 (JIT isitish bilan)", flush=True)
    refs = {}
    for acc in ("f64", "f32"):
        run_chain(X, args, 1, "det", acc, args.block, bounds)  # isitish
        w0, S0, _ = run_chain(X, args, 1, "det", acc, args.block, bounds)
        refs[acc] = dict(w=w0, top=topk_masks(S0, args.topk), cls=classes(S0, args.breaks))
        env[f"w_ref_{acc}"] = w0.tolist()
        env[f"w_ref_{acc}_hash"] = sha(w0)
        env[f"S_ref_{acc}_hash"] = sha(S0)
    w_ref = refs["f64"]["w"]

    configs = []
    for acc in ("f64", "f32"):
        for T in args.threads_det:
            configs.append(("P", "det", acc, T, args.block))
    for acc in ("f64", "f32"):
        for T in args.threads:
            configs.append(("A3", "naive", acc, T, args.block))
    for bb in args.blocks_alt:
        configs.append(("P_b", "det", "f64", max(args.threads_det), bb))

    rows = []
    fields = ["variant", "mode", "acc", "T", "block", "run", "seconds", "w_hash", "S_hash",
              "max_abs_dw", "max_abs_dw_vs_f64", "n_class_changed", "n_class_changed_vs_f64"] \
        + [f"jaccard_top{f}" for f in args.topk] + [f"jaccard_top{f}_vs_f64" for f in args.topk]
    for (var, mode, acc, T, bb) in configs:
        reps = args.runs if var != "P_b" else 3
        print(f"{var} mode={mode} acc={acc} T={T} b={bb}: {reps} ta", flush=True)
        for r in range(reps):
            w, S, sec = run_chain(X, args, T, mode, acc, bb, bounds)
            top = topk_masks(S, args.topk)
            ref = refs[acc]
            row = dict(variant=var, mode=mode, acc=acc, T=T, block=bb, run=r, seconds=round(sec, 4),
                       w_hash=sha(w), S_hash=sha(S),
                       max_abs_dw=float(np.max(np.abs(w - ref["w"]))),
                       max_abs_dw_vs_f64=float(np.max(np.abs(w - w_ref))),
                       n_class_changed=int(np.count_nonzero(classes(S, args.breaks) != ref["cls"])),
                       n_class_changed_vs_f64=int(np.count_nonzero(classes(S, args.breaks) != refs["f64"]["cls"])))
            for f in args.topk:
                row[f"jaccard_top{f}"] = jaccard(top[f], ref["top"][f])
                row[f"jaccard_top{f}_vs_f64"] = jaccard(top[f], refs["f64"]["top"][f])
            rows.append(row)

    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    for (var, mode, acc, T, bb) in configs:
        g = [r for r in rows if (r["variant"], r["acc"], r["T"], r["block"]) == (var, acc, T, bb)]
        summary.append(dict(
            variant=var, acc=acc, T=T, block=bb, runs=len(g),
            distinct_w=len({r["w_hash"] for r in g}),
            distinct_S=len({r["S_hash"] for r in g}),
            max_abs_dw=max(r["max_abs_dw"] for r in g),
            max_abs_dw_vs_f64=max(r["max_abs_dw_vs_f64"] for r in g),
            max_class_changed=max(r["n_class_changed"] for r in g),
            max_class_changed_vs_f64=max(r["n_class_changed_vs_f64"] for r in g),
            **{f"min_jaccard_top{f}_vs_f64": float(min(r[f"jaccard_top{f}_vs_f64"] for r in g)) for f in args.topk},
            **{f"min_jaccard_top{f}": float(min(r[f"jaccard_top{f}"] for r in g)) for f in args.topk},
            median_seconds=float(np.median([r["seconds"] for r in g]))))
    json.dump(dict(env=env, summary=summary), open(os.path.join(args.out, "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
