#!/usr/bin/env python3
"""
E1 — O'tishlar va yadrolar hissasi (1-maqola, IJGI protokoli).

Zanjir (barcha variantlarda bir xil): persentil normallashtirish (alpha), EWM vaznlari,
SAW, TOPSIS, VIKOR (v = 0.5). Xaritalar S (SAW), C (TOPSIS), Q (VIKOR) diskka yoziladi.

Variantlar:
  B0 : normallashtirilgan matritsa butunlay xotirada (NumPy), usullar alohida
  B1 : 9 o'tish, bloklab, NumPy massiv amallari
  A1 : 4 + 1 o'tish, bloklab, NumPy massiv amallari
  A2 : 9 o'tish, bloklab, birlashtirilgan Numba yadrolari
  P  : 4 + 1 o'tish, bloklab, birlashtirilgan Numba yadrolari (taklif)
  P5 : 5 to'liq o'tish, Numba; S^V, R diskka yozilmaydi, VIKOR Q ma'lumotdan qayta
       hisoblanadi (keshdan tashqari rejimda yozish narxini ajratish uchun)

9 o'tish (III bob): statistika; gistogramma; EWM ustun yig'indilari; EWM ulushlar va
entropiya; SAW; TOPSIS ideal nuqtalari (ma'lumotdan); TOPSIS C; VIKOR S, R ekstremumlari;
VIKOR Q (ma'lumotdan qayta).
4 + 1 o'tish (Teorema 3.1): statistika; gistogramma; R_j, Q_j (Teorema 2.1); S, C, S^V, R
xaritalari + ekstremumlar (TOPSIS qat'iy ideal nuqtalar bilan); yengil o'tish: Q yozilgan
S^V, R dan.

Barcha reduksiyalar blok raqami tartibida birlashtiriladi (deterministik).

Misol:
  python e1_passes.py --data data\\orol_v2_100m\\data.npy --dirs + - - + - + - - ^
      --variants B0 B1 A1 A2 P --threads 1 2 4 6 8 12 --repeats 10 --out experiments\\e1\\smoke_100m
"""
import argparse, csv, hashlib, json, os, platform, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import numba
from numba import njit

try:
    import psutil
except ImportError:  # xotira/I/O o'lchovi bo'lmaydi
    psutil = None

V_VIKOR = 0.5

# =========================================================== Numba yadrolari
# fastmath=False (standart)


@njit(nogil=True, cache=True)
def _ok(X, i, nodata):
    for j in range(X.shape[0]):
        v = X[j, i]
        if not np.isfinite(v) or v == nodata:
            return False
    return True


@njit(nogil=True, cache=True)
def _norm(x, L, U, d):
    if U <= L:
        return 0.0
    r = (x - L) / (U - L) if d > 0 else (U - x) / (U - L)
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


@njit(nogil=True, cache=True)
def f_stats(X, nodata):
    n, b = X.shape
    cnt = 0
    mn = np.full(n, np.inf)
    mx = np.full(n, -np.inf)
    for i in range(b):
        if not _ok(X, i, nodata):
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
def f_hist(X, nodata, mn, mx, nb):
    n, b = X.shape
    H = np.zeros((n, nb), np.int64)
    for i in range(b):
        if not _ok(X, i, nodata):
            continue
        for j in range(n):
            w = mx[j] - mn[j]
            h = 0
            if w > 0.0:
                h = int((np.float64(X[j, i]) - mn[j]) / w * nb)
                h = min(max(h, 0), nb - 1)
            H[j, h] += 1
    return H


@njit(nogil=True, cache=True)
def f_rq(X, nodata, L, U, dirs, c):
    """P/A2 emas: P ning 3-o'tishi — R_j va Q_j bitta o'tishda (Teorema 2.1)."""
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
            if not _ok(X, i, nodata):
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
def f_colsum(X, nodata, L, U, dirs, c):
    """A2: EWM 1-o'tishi — ustun yig'indilari."""
    n, b = X.shape
    R = np.zeros(n)
    rho = np.zeros(n)
    s = 0
    while s < b:
        e = min(s + c, b)
        rho[:] = 0.0
        for i in range(s, e):
            if not _ok(X, i, nodata):
                continue
            for j in range(n):
                rho[j] += _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
        for j in range(n):
            R[j] += rho[j]
        s = e
    return R


@njit(nogil=True, cache=True)
def f_entropy(X, nodata, L, U, dirs, Rtot, c):
    """A2: EWM 2-o'tishi — sum p ln p, p = r / R_j."""
    n, b = X.shape
    E = np.zeros(n)
    th = np.zeros(n)
    s = 0
    while s < b:
        e = min(s + c, b)
        th[:] = 0.0
        for i in range(s, e):
            if not _ok(X, i, nodata):
                continue
            for j in range(n):
                if Rtot[j] > 0.0:
                    p = _norm(np.float64(X[j, i]), L[j], U[j], dirs[j]) / Rtot[j]
                    if p > 0.0:
                        th[j] += p * np.log(p)
        for j in range(n):
            E[j] += th[j]
        s = e
    return E


@njit(nogil=True, cache=True)
def f_saw(X, nodata, L, U, dirs, w, out):
    n, b = X.shape
    for i in range(b):
        if not _ok(X, i, nodata):
            out[i] = np.nan
            continue
        s = 0.0
        for j in range(n):
            s += w[j] * _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
        out[i] = s


@njit(nogil=True, cache=True)
def f_topsis_ideal(X, nodata, L, U, dirs, w):
    n, b = X.shape
    vmax = np.full(n, -np.inf)
    vmin = np.full(n, np.inf)
    for i in range(b):
        if not _ok(X, i, nodata):
            continue
        for j in range(n):
            v = w[j] * _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
            if v > vmax[j]:
                vmax[j] = v
            if v < vmin[j]:
                vmin[j] = v
    return vmax, vmin


@njit(nogil=True, cache=True)
def f_topsis_c(X, nodata, L, U, dirs, w, vmax, vmin, out):
    n, b = X.shape
    for i in range(b):
        if not _ok(X, i, nodata):
            out[i] = np.nan
            continue
        dp = 0.0
        dm = 0.0
        for j in range(n):
            v = w[j] * _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
            dp += (vmax[j] - v) ** 2
            dm += (v - vmin[j]) ** 2
        dp = np.sqrt(dp)
        dm = np.sqrt(dm)
        out[i] = dm / (dp + dm) if dp + dm > 0.0 else np.nan


@njit(nogil=True, cache=True)
def f_vikor_ext(X, nodata, L, U, dirs, w):
    n, b = X.shape
    ext = np.array([np.inf, -np.inf, np.inf, -np.inf])  # Smin Smax Rmin Rmax
    for i in range(b):
        if not _ok(X, i, nodata):
            continue
        sv = 0.0
        rr = 0.0
        for j in range(n):
            t = w[j] * (1.0 - _norm(np.float64(X[j, i]), L[j], U[j], dirs[j]))
            sv += t
            if t > rr:
                rr = t
        ext[0] = min(ext[0], sv)
        ext[1] = max(ext[1], sv)
        ext[2] = min(ext[2], rr)
        ext[3] = max(ext[3], rr)
    return ext


@njit(nogil=True, cache=True)
def _q(sv, rr, ext, v):
    a = (sv - ext[0]) / (ext[1] - ext[0]) if ext[1] > ext[0] else 0.0
    bq = (rr - ext[2]) / (ext[3] - ext[2]) if ext[3] > ext[2] else 0.0
    return v * a + (1.0 - v) * bq


@njit(nogil=True, cache=True)
def f_vikor_q(X, nodata, L, U, dirs, w, ext, v, out):
    n, b = X.shape
    for i in range(b):
        if not _ok(X, i, nodata):
            out[i] = np.nan
            continue
        sv = 0.0
        rr = 0.0
        for j in range(n):
            t = w[j] * (1.0 - _norm(np.float64(X[j, i]), L[j], U[j], dirs[j]))
            sv += t
            if t > rr:
                rr = t
        out[i] = _q(sv, rr, ext, v)


@njit(nogil=True, cache=True)
def f_scores(X, nodata, L, U, dirs, w, oS, oC, oSV, oR):
    """P ning 4-o'tishi: S, C (qat'iy ideal: v+ = w_j, v- = 0), S^V, R + ekstremumlar."""
    n, b = X.shape
    ext = np.array([np.inf, -np.inf, np.inf, -np.inf])
    for i in range(b):
        if not _ok(X, i, nodata):
            oS[i] = np.nan
            oC[i] = np.nan
            oSV[i] = np.nan
            oR[i] = np.nan
            continue
        s = 0.0
        dp = 0.0
        dm = 0.0
        sv = 0.0
        rr = 0.0
        for j in range(n):
            r = _norm(np.float64(X[j, i]), L[j], U[j], dirs[j])
            v = w[j] * r
            s += v
            dp += (w[j] - v) ** 2
            dm += v * v
            t = w[j] - v
            sv += t
            if t > rr:
                rr = t
        dp = np.sqrt(dp)
        dm = np.sqrt(dm)
        oS[i] = s
        oC[i] = dm / (dp + dm) if dp + dm > 0.0 else np.nan
        oSV[i] = sv
        oR[i] = rr
        ext[0] = min(ext[0], sv)
        ext[1] = max(ext[1], sv)
        ext[2] = min(ext[2], rr)
        ext[3] = max(ext[3], rr)
    return ext


@njit(nogil=True, cache=True)
def f_light_q(SV, R, ext, v, out):
    for i in range(SV.shape[0]):
        sv = SV[i]
        if not np.isfinite(sv):
            out[i] = np.nan
        else:
            out[i] = _q(np.float64(sv), np.float64(R[i]), ext, v)

# =========================================================== NumPy blok funksiyalari


def n_valid(Xb, nodata):
    ok = np.isfinite(Xb).all(axis=0)
    if np.isfinite(nodata):
        ok &= (Xb != nodata).all(axis=0)
    return ok


def n_norm(Xv, L, U, dirs):
    """Xv: (n, m_b) float32 -> r: (n, m_b) float64 (oraliq massiv)."""
    x = Xv.astype(np.float64)
    den = (U - L)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dirs[:, None] > 0, (x - L[:, None]) / den, (U[:, None] - x) / den)
    r = np.clip(r, 0.0, 1.0)
    r[U <= L, :] = 0.0
    return r


def n_stats(Xb, nodata):
    ok = n_valid(Xb, nodata)
    Xv = Xb[:, ok].astype(np.float64)
    n = Xb.shape[0]
    if Xv.shape[1] == 0:
        return 0, np.full(n, np.inf), np.full(n, -np.inf)
    return int(ok.sum()), Xv.min(axis=1), Xv.max(axis=1)


def n_hist(Xb, nodata, mn, mx, nb):
    ok = n_valid(Xb, nodata)
    Xv = Xb[:, ok].astype(np.float64)
    n = Xb.shape[0]
    H = np.zeros((n, nb), np.int64)
    for j in range(n):
        w = mx[j] - mn[j]
        if w > 0:
            h = ((Xv[j] - mn[j]) / w * nb).astype(np.int64)
            h = np.clip(h, 0, nb - 1)
        else:
            h = np.zeros(Xv.shape[1], np.int64)
        H[j] = np.bincount(h, minlength=nb)
    return H


def n_rq(Xb, nodata, L, U, dirs):
    r = n_norm(Xb[:, n_valid(Xb, nodata)], L, U, dirs)
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = np.where(r > 0, r * np.log(r), 0.0)
    return r.sum(axis=1), rl.sum(axis=1)


def n_colsum(Xb, nodata, L, U, dirs):
    return n_norm(Xb[:, n_valid(Xb, nodata)], L, U, dirs).sum(axis=1)


def n_entropy(Xb, nodata, L, U, dirs, Rtot):
    r = n_norm(Xb[:, n_valid(Xb, nodata)], L, U, dirs)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(Rtot[:, None] > 0, r / Rtot[:, None], 0.0)
        pl = np.where(p > 0, p * np.log(p), 0.0)
    return pl.sum(axis=1)


def _scatter(out, ok, vals):
    out[:] = np.nan
    out[ok] = vals


def n_saw(Xb, nodata, L, U, dirs, w, out):
    ok = n_valid(Xb, nodata)
    r = n_norm(Xb[:, ok], L, U, dirs)
    _scatter(out, ok, (w[:, None] * r).sum(axis=0))


def n_topsis_ideal(Xb, nodata, L, U, dirs, w):
    v = w[:, None] * n_norm(Xb[:, n_valid(Xb, nodata)], L, U, dirs)
    n = Xb.shape[0]
    if v.shape[1] == 0:
        return np.full(n, -np.inf), np.full(n, np.inf)
    return v.max(axis=1), v.min(axis=1)


def n_topsis_c(Xb, nodata, L, U, dirs, w, vmax, vmin, out):
    ok = n_valid(Xb, nodata)
    v = w[:, None] * n_norm(Xb[:, ok], L, U, dirs)
    dp = np.sqrt(((vmax[:, None] - v) ** 2).sum(axis=0))
    dm = np.sqrt(((v - vmin[:, None]) ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        _scatter(out, ok, dm / (dp + dm))


def _sv_r(t):
    return t.sum(axis=0), t.max(axis=0)


def n_vikor_ext(Xb, nodata, L, U, dirs, w):
    t = w[:, None] * (1.0 - n_norm(Xb[:, n_valid(Xb, nodata)], L, U, dirs))
    if t.shape[1] == 0:
        return np.array([np.inf, -np.inf, np.inf, -np.inf])
    sv, rr = _sv_r(t)
    return np.array([sv.min(), sv.max(), rr.min(), rr.max()])


def n_qvals(sv, rr, ext, v):
    a = (sv - ext[0]) / (ext[1] - ext[0]) if ext[1] > ext[0] else np.zeros_like(sv)
    b = (rr - ext[2]) / (ext[3] - ext[2]) if ext[3] > ext[2] else np.zeros_like(rr)
    return v * a + (1.0 - v) * b


def n_vikor_q(Xb, nodata, L, U, dirs, w, ext, v, out):
    ok = n_valid(Xb, nodata)
    t = w[:, None] * (1.0 - n_norm(Xb[:, ok], L, U, dirs))
    sv, rr = _sv_r(t)
    _scatter(out, ok, n_qvals(sv, rr, ext, v))


def n_scores(Xb, nodata, L, U, dirs, w, oS, oC, oSV, oR):
    ok = n_valid(Xb, nodata)
    r = n_norm(Xb[:, ok], L, U, dirs)
    v = w[:, None] * r
    s = v.sum(axis=0)
    dp = np.sqrt(((w[:, None] - v) ** 2).sum(axis=0))
    dm = np.sqrt((v ** 2).sum(axis=0))
    t = w[:, None] - v
    sv, rr = _sv_r(t)
    _scatter(oS, ok, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        _scatter(oC, ok, dm / (dp + dm))
    _scatter(oSV, ok, sv)
    _scatter(oR, ok, rr)
    if sv.size == 0:
        return np.array([np.inf, -np.inf, np.inf, -np.inf])
    return np.array([sv.min(), sv.max(), rr.min(), rr.max()])


def n_light_q(SV, R, ext, v, out):
    sv = SV.astype(np.float64)
    rr = R.astype(np.float64)
    out[:] = n_qvals(sv, rr, ext, v)  # NaN tarqaladi

# =========================================================== umumiy yordamchilar


def blocks(K, b):
    return [(lo, min(lo + b, K)) for lo in range(0, K, b)]


def hist_quantile(Hj, mn, mx, m, q):
    nb = Hj.shape[0]
    if mx <= mn:
        return mn
    C = np.cumsum(Hj)
    t = q * m
    h = min(int(np.searchsorted(C, t, side="left")), nb - 1)
    prev = C[h - 1] if h > 0 else 0
    frac = (t - prev) / Hj[h] if Hj[h] > 0 else 0.0
    return mn + (h + frac) * (mx - mn) / nb


def bounds_from(H, mn, mx, m, alpha):
    n = H.shape[0]
    L = np.array([hist_quantile(H[j], mn[j], mx[j], m, alpha) for j in range(n)])
    U = np.array([hist_quantile(H[j], mn[j], mx[j], m, 1 - alpha) for j in range(n)])
    return L, U


def w_from_rq(R, Q, m, L, U):
    n = R.shape[0]
    d = np.zeros(n)
    for j in range(n):
        if U[j] > L[j] and R[j] > 0:
            d[j] = 1.0 - (np.log(R[j]) - Q[j] / R[j]) / np.log(m)
    return d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)


def w_from_entropy(E, m, L, U):
    n = E.shape[0]
    d = np.zeros(n)
    for j in range(n):
        if U[j] > L[j]:
            d[j] = 1.0 - (-E[j] / np.log(m))
    return d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)


def sha(a, nbytes=None):
    h = hashlib.sha256()
    a = np.asarray(a)
    step = 1 << 24
    flat = a.reshape(-1)
    for s in range(0, flat.size, step):
        h.update(np.ascontiguousarray(flat[s:s + step]).tobytes())
    return h.hexdigest()[:16]


class MemSampler:
    """Jarayon xotirasini 50 ms qadamda kuzatadi.

    rss     — ishchi to'plam; Windows da xotiraga akslantirilgan fayl sahifalarini
              (data.npy, xaritalar) ham o'z ichiga oladi, shuning uchun Lemma 3.3 ni
              tekshirish uchun yaroqsiz.
    private — jarayonning shaxsiy (fayl bilan bog'lanmagan) xotirasi: Windows da
              memory_info().private, boshqa tizimlarda uss (memory_full_info).
    """

    def __init__(self, dt=0.05):
        self.dt, self.peak, self.peak_private, self._stop = dt, 0, 0, threading.Event()
        self.p = psutil.Process() if psutil else None

    def _sample(self):
        mi = self.p.memory_info()
        self.peak = max(self.peak, mi.rss)
        pv = getattr(mi, "private", None)
        if pv is None:
            try:
                pv = self.p.memory_full_info().uss
            except Exception:
                pv = 0
        self.peak_private = max(self.peak_private, pv)

    def __enter__(self):
        if self.p:
            self._sample()
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.dt)

    def __exit__(self, *a):
        if self.p:
            self._stop.set()
            self.t.join()
            self._sample()


def private_mb():
    if not psutil:
        return None
    p = psutil.Process()
    pv = getattr(p.memory_info(), "private", None)
    if pv is None:
        try:
            pv = p.memory_full_info().uss
        except Exception:
            return None
    return round(pv / 2 ** 20, 1)


def disk_bytes():
    """Butun tizim bo'yicha disk o'qish/yozish baytlari (psutil.disk_io_counters).
    Jarayon hisoblagichi (io_counters) Windows da xotiraga akslantirilgan fayl
    sahifalarini o'qishni hisoblamaydi, shuning uchun tizim hisoblagichi olinadi;
    unga fon jarayonlarining I/O si ham kiradi."""
    if not psutil:
        return None
    try:
        c = psutil.disk_io_counters()
        return c.read_bytes, c.write_bytes
    except Exception:
        return None

# =========================================================== variantlar


class Ctx:
    pass


def _map(pool, fn, items):
    """Bloklar parallel, natijalar blok raqami tartibida qaytariladi."""
    return [f.result() for f in [pool.submit(_call, fn, it) for it in items]]


class _Lazy:
    """Diskka yozilmaydigan xarita bo'lagi: bufer faqat yadro bajarilayotganda
    (ishchi oqim ichida) ajratiladi, shuning uchun bir vaqtda ko'pi bilan T ta bo'ladi."""

    def __init__(self, n):
        self.n = n


class _Scratch:
    def __getitem__(self, sl):
        return _Lazy(sl.stop - sl.start)

    def flush(self):
        pass


def _call(fn, args):
    return fn(*[np.empty(a.n, np.float32) if isinstance(a, _Lazy) else a for a in args])


def _outs(ctx, variant):
    """write = all: barcha xaritalar diskka; needed: faqat algoritm uchun zarur
    xaritalar (4 + 1 sxemada S^V va R), qolganlari vaqtinchalik buferga."""
    M = ctx.maps[variant]
    keys = ("S", "C", "SV", "R", "Q")
    if ctx.write == "all":
        return {k: M.get(k, _Scratch()) for k in keys}
    return {k: (M[k] if (k in ("SV", "R") and k in M) else _Scratch()) for k in keys}


def run_blocked(ctx, variant, T):
    """B1, A1, A2, P. Qaytaradi: (w, o'tishlar vaqtlari dict)."""
    X, blk, nd, dirs, c = ctx.X, ctx.blk, ctx.nodata, ctx.dirs, ctx.chunk
    fused = variant in ("A2", "P", "P5")
    nine = variant in ("B1", "A2")
    M = _outs(ctx, variant)
    t = {}
    with ThreadPoolExecutor(max_workers=T) as pool:
        # 1 statistika
        t0 = time.perf_counter()
        fn = f_stats if fused else n_stats
        res = _map(pool, fn, [(X[:, lo:hi], nd) for lo, hi in blk])
        m = sum(r[0] for r in res)
        mn = np.min(np.stack([r[1] for r in res]), axis=0)
        mx = np.max(np.stack([r[2] for r in res]), axis=0)
        t["p1_stats"] = time.perf_counter() - t0
        # 2 gistogramma -> L, U
        t0 = time.perf_counter()
        fn = f_hist if fused else n_hist
        # Butun sonli gistogramma: yig'indi aniq va tartibga bog'liq emas (Lemma 3.2),
        # shuning uchun blok natijalari tayyor bo'lishi bilan qo'shiladi va xotirada
        # P ta gistogramma emas, ko'pi bilan ~T ta saqlanadi.
        H = np.zeros((X.shape[0], ctx.nbins), np.int64)
        futs = set()
        for lo, hi in blk:
            futs.add(pool.submit(fn, X[:, lo:hi], nd, mn, mx, ctx.nbins))
            if len(futs) >= 2 * T:
                done, futs = wait(futs, return_when=FIRST_COMPLETED)
                for f in done:
                    H += f.result()
        for f in futs:
            H += f.result()
        L, U = bounds_from(H, mn, mx, m, ctx.alpha)
        t["p2_hist"] = time.perf_counter() - t0
        if nine:
            # 3-4 EWM ikki o'tishda
            t0 = time.perf_counter()
            if fused:
                parts = _map(pool, f_colsum, [(X[:, lo:hi], nd, L, U, dirs, c) for lo, hi in blk])
            else:
                parts = _map(pool, n_colsum, [(X[:, lo:hi], nd, L, U, dirs) for lo, hi in blk])
            R = np.zeros(X.shape[0])
            for p in parts:
                R += p
            t["p3_ewm_colsum"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            if fused:
                parts = _map(pool, f_entropy, [(X[:, lo:hi], nd, L, U, dirs, R, c) for lo, hi in blk])
            else:
                parts = _map(pool, n_entropy, [(X[:, lo:hi], nd, L, U, dirs, R) for lo, hi in blk])
            E = np.zeros(X.shape[0])
            for p in parts:
                E += p
            w = w_from_entropy(E, m, L, U)
            t["p4_ewm_entropy"] = time.perf_counter() - t0
            # 5 SAW
            t0 = time.perf_counter()
            fn = f_saw if fused else n_saw
            _map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w, M["S"][lo:hi]) for lo, hi in blk])
            t["p5_saw"] = time.perf_counter() - t0
            # 6-7 TOPSIS
            t0 = time.perf_counter()
            fn = f_topsis_ideal if fused else n_topsis_ideal
            res = _map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w) for lo, hi in blk])
            vmax = np.max(np.stack([r[0] for r in res]), axis=0)
            vmin = np.min(np.stack([r[1] for r in res]), axis=0)
            t["p6_topsis_ideal"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn = f_topsis_c if fused else n_topsis_c
            _map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w, vmax, vmin, M["C"][lo:hi]) for lo, hi in blk])
            t["p7_topsis_c"] = time.perf_counter() - t0
            # 8-9 VIKOR
            t0 = time.perf_counter()
            fn = f_vikor_ext if fused else n_vikor_ext
            res = np.stack(_map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w) for lo, hi in blk]))
            ext = np.array([res[:, 0].min(), res[:, 1].max(), res[:, 2].min(), res[:, 3].max()])
            t["p8_vikor_ext"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn = f_vikor_q if fused else n_vikor_q
            _map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w, ext, V_VIKOR, M["Q"][lo:hi]) for lo, hi in blk])
            t["p9_vikor_q"] = time.perf_counter() - t0
        else:
            # 3 R_j, Q_j bitta o'tishda
            t0 = time.perf_counter()
            if fused:
                parts = _map(pool, f_rq, [(X[:, lo:hi], nd, L, U, dirs, c) for lo, hi in blk])
            else:
                parts = _map(pool, n_rq, [(X[:, lo:hi], nd, L, U, dirs) for lo, hi in blk])
            R = np.zeros(X.shape[0])
            Q = np.zeros(X.shape[0])
            for r, q in parts:
                R += r
                Q += q
            w = w_from_rq(R, Q, m, L, U)
            t["p3_ewm_rq"] = time.perf_counter() - t0
            # 4 S, C, S^V, R + ekstremumlar
            t0 = time.perf_counter()
            fn = f_scores if fused else n_scores
            res = np.stack(_map(pool, fn, [(X[:, lo:hi], nd, L, U, dirs, w, M["S"][lo:hi], M["C"][lo:hi],
                                            M["SV"][lo:hi], M["R"][lo:hi]) for lo, hi in blk]))
            ext = np.array([res[:, 0].min(), res[:, 1].max(), res[:, 2].min(), res[:, 3].max()])
            t["p4_scores"] = time.perf_counter() - t0
            if variant == "P5":
                # 5 to'liq o'tish: S^V, R saqlanmaydi, Q dastlabki ma'lumotdan qayta hisoblanadi
                t0 = time.perf_counter()
                _map(pool, f_vikor_q, [(X[:, lo:hi], nd, L, U, dirs, w, ext, V_VIKOR, M["Q"][lo:hi])
                                       for lo, hi in blk])
                t["p5_full_q"] = time.perf_counter() - t0
            else:
                # 5 yengil o'tish: Q yozilgan S^V, R dan
                t0 = time.perf_counter()
                fn = f_light_q if fused else n_light_q
                _map(pool, fn, [(M["SV"][lo:hi], M["R"][lo:hi], ext, V_VIKOR, M["Q"][lo:hi]) for lo, hi in blk])
                t["p5_light_q"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for a in M.values():
        a.flush()
    t["flush"] = time.perf_counter() - t0
    return w, t


def run_b0(ctx):
    """Normallashtirilgan matritsa xotirada; usullar alohida, NumPy."""
    X, nd, dirs = ctx.X, ctx.nodata, ctx.dirs
    n, K = X.shape
    M = ctx.maps["B0"]
    t = {}
    t0 = time.perf_counter()
    ok = np.ones(K, bool)
    for j in range(n):
        xj = np.asarray(X[j])
        ok &= np.isfinite(xj)
        if np.isfinite(nd):
            ok &= xj != nd
    m = int(ok.sum())
    Xv = np.empty((n, m), np.float32)
    for j in range(n):
        Xv[j] = X[j][ok]
    t["load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mn = Xv.min(axis=1).astype(np.float64)
    mx = Xv.max(axis=1).astype(np.float64)
    H = np.zeros((n, ctx.nbins), np.int64)
    for j in range(n):
        w_ = mx[j] - mn[j]
        h = np.clip(((Xv[j].astype(np.float64) - mn[j]) / w_ * ctx.nbins).astype(np.int64), 0, ctx.nbins - 1) \
            if w_ > 0 else np.zeros(m, np.int64)
        H[j] = np.bincount(h, minlength=ctx.nbins)
    L, U = bounds_from(H, mn, mx, m, ctx.alpha)
    t["bounds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    N = np.empty((n, m), np.float32)  # normallashtirilgan matritsa
    for j in range(n):
        N[j] = n_norm(Xv[j:j + 1], L[j:j + 1], U[j:j + 1], dirs[j:j + 1])[0]
    del Xv
    E = np.zeros(n)
    for j in range(n):
        r = N[j].astype(np.float64)
        Rj = r.sum()
        if Rj > 0:
            p = r / Rj
            E[j] = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum()
    w = w_from_entropy(E, m, L, U)
    t["norm_ewm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    S = np.zeros(m)
    for j in range(n):
        S += w[j] * N[j]
    vmax = np.array([w[j] * N[j].max() for j in range(n)])
    vmin = np.array([w[j] * N[j].min() for j in range(n)])
    dp = np.zeros(m)
    dm = np.zeros(m)
    SV = np.zeros(m)
    R = np.zeros(m)
    for j in range(n):
        v = w[j] * N[j].astype(np.float64)
        dp += (vmax[j] - v) ** 2
        dm += (v - vmin[j]) ** 2
        tj = w[j] - v
        SV += tj
        np.maximum(R, tj, out=R)
    dp = np.sqrt(dp)
    dm = np.sqrt(dm)
    C = dm / (dp + dm)
    ext = np.array([SV.min(), SV.max(), R.min(), R.max()])
    Qv = n_qvals(SV, R, ext, V_VIKOR)
    t["methods"] = time.perf_counter() - t0
    if ctx.write == "all":
        t0 = time.perf_counter()
        for key, vals in (("S", S), ("C", C), ("Q", Qv)):
            M[key][:] = np.nan
            M[key][ok] = vals
            M[key].flush()
        t["write"] = time.perf_counter() - t0
    return w, t


def run_variant(ctx, variant, T):
    if variant == "B0":
        return run_b0(ctx)
    return run_blocked(ctx, variant, T)


def open_maps(out, variant, K, keys):
    d = os.path.join(out, "maps", variant)
    os.makedirs(d, exist_ok=True)
    return {k: np.lib.format.open_memmap(os.path.join(d, f"{k}.npy"), mode="w+", dtype=np.float32, shape=(K,))
            for k in keys}


def max_diff(a, b, step=1 << 24):
    """Blokma-blok max |a - b| (NaN o'rinlari mos kelishi ham tekshiriladi)."""
    md, nan_mismatch = 0.0, 0
    for s in range(0, a.shape[0], step):
        x = np.asarray(a[s:s + step], np.float64)
        y = np.asarray(b[s:s + step], np.float64)
        fx, fy = np.isfinite(x), np.isfinite(y)
        nan_mismatch += int(np.count_nonzero(fx != fy))
        both = fx & fy
        if both.any():
            md = max(md, float(np.max(np.abs(x[both] - y[both]))))
    return md, nan_mismatch

# =========================================================== main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".npy, shakli (n, K), float32")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--nodata", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--block", type=int, default=2 ** 20)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--nbins", type=int, default=65536)
    ap.add_argument("--variants", nargs="+", default=["B0", "B1", "A1", "A2", "P"])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12],
                    help="P uchun (--threads_all bilan B1, A1, A2 uchun ham)")
    ap.add_argument("--threads_all", action="store_true",
                    help="--threads ro'yxatini B1, A1, A2 ga ham qo'llash (B0 doim 1 oqim)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--pause", type=float, default=2.0, help="ishga tushirishlar orasidagi tanaffus, s")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--map_tol", type=float, default=1e-5, help="xarita farqi chegarasi (to'g'rilik)")
    ap.add_argument("--write", choices=["all", "needed"], default="needed",
                    help="o'lchanadigan ishga tushirishlarda: all — S, C, Q xaritalari ham diskka; "
                         "needed — faqat 4 + 1 sxema uchun zarur S^V, R (tekshiruv doim 'all' bilan)")
    ap.add_argument("--no_check", action="store_true",
                    help="to'g'rilik tekshiruvisiz: faqat 4 + 1 uchun zarur S^V, R fayllari yaratiladi "
                         "(xotiradan katta ma'lumot uchun disk joyini tejaydi)")
    ap.add_argument("--out", default="e1_out")
    args = ap.parse_args()

    X = np.load(args.data, mmap_mode="r")
    n, K = X.shape
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.data)), "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if args.nodata is None:
        args.nodata = float(meta["nodata"]) if meta.get("nodata") is not None else np.nan
    if len(args.dirs) != n:
        sys.exit(f"--dirs da {len(args.dirs)} ta belgi, n = {n}")
    names = meta.get("names") if meta.get("names") and len(meta["names"]) == n else [f"mezon_{j+1}" for j in range(n)]
    os.makedirs(args.out, exist_ok=True)

    base_import_mb = private_mb()  # import'dan keyin, hisobdan oldin
    ctx = Ctx()
    ctx.X, ctx.nodata, ctx.alpha, ctx.chunk, ctx.nbins = X, args.nodata, args.alpha, args.chunk, args.nbins
    ctx.dirs = np.array([1 if d == "+" else -1 for d in args.dirs], np.int64)
    ctx.blk = blocks(K, args.block)
    ctx.write = "all"  # etalon va tekshiruv doim to'liq yozish bilan
    print(f"fayl: {args.data}; (n, K) = {X.shape}; bloklar P = {len(ctx.blk)}; nodata = {args.nodata}", flush=True)
    print("  mezonlar:", ", ".join(f"{a} ({d})" for a, d in zip(names, args.dirs)), flush=True)
    if psutil is None:
        print("OGOHLANTIRISH: psutil yo'q — xotira va I/O yozilmaydi (pip install psutil)", flush=True)

    ctx.maps = {}
    for v in args.variants:
        if args.no_check:
            keys = ("SV", "R") if v in ("A1", "P") else ()
        else:
            keys = ("S", "C", "SV", "R", "Q") if v in ("A1", "P") else ("S", "C", "Q")
        ctx.maps[v] = open_maps(args.out, v, K, keys)
    ctx.maps["REF"] = open_maps(args.out, "REF", K, ("SV", "R") if args.no_check else ("S", "C", "SV", "R", "Q"))
    if args.no_check:
        ctx.write = "needed"
        if "B0" in args.variants:
            sys.exit("--no_check bilan B0 ishlatilmaydi")

    # Etalon: P, T = 1 (JIT isitish bilan), alohida xaritalarga
    print("etalon: P, T = 1", flush=True)
    saved = ctx.maps.get("P")
    ctx.maps["P"] = ctx.maps["REF"]
    run_blocked(ctx, "P", 1)
    w_ref, _ = run_blocked(ctx, "P", 1)
    ctx.maps["P"] = saved if saved is not None else ctx.maps["REF"]
    if saved is None:
        del ctx.maps["P"]

    configs = []
    for v in args.variants:
        for T in (args.threads if (v in ("P", "P5") or (args.threads_all and v != "B0")) else [1]):
            configs.append((v, T))

    # Isitish va to'g'rilik tekshiruvi (o'lchanmaydi)
    check = {}
    for v, T in configs:
        if args.no_check:
            print(f"isitish (tekshiruvsiz): {v} T={T}", flush=True)
            w, _ = run_variant(ctx, v, T)
            check[(v, T)] = dict(status="tekshirilmadi", max_abs_dw=float(np.max(np.abs(w - w_ref))))
            continue
        print(f"isitish va tekshiruv: {v} T={T}", flush=True)
        try:
            w, _ = run_variant(ctx, v, T)
        except MemoryError:
            check[(v, T)] = dict(status="xotira yetmadi")
            print("  xotira yetmadi", flush=True)
            continue
        d = dict(status="ok", max_abs_dw=float(np.max(np.abs(w - w_ref))))
        for k in ("S", "C", "Q"):
            md, nm = max_diff(ctx.maps[v][k], ctx.maps["REF"][k])
            d[f"max_abs_d{k}"], d[f"nan_mismatch_{k}"] = md, nm
        if any(d[f"max_abs_d{k}"] > args.map_tol or d[f"nan_mismatch_{k}"] > 0 for k in ("S", "C", "Q")):
            d["status"] = "FARQ"
        check[(v, T)] = d
        print("  ", {k: (f"{x:.2e}" if isinstance(x, float) else x) for k, x in d.items()}, flush=True)

    base_after_jit_mb = private_mb()  # JIT kompilyatsiya va tekshiruvlardan keyin
    print(f"jarayon shaxsiy xotirasi: import'dan keyin {base_import_mb} MB, "
          f"JIT va tekshiruvdan keyin {base_after_jit_mb} MB", flush=True)
    # O'lchov: tasodifiy tartib
    ctx.write = args.write
    jobs = [(v, T, r) for (v, T) in configs if check[(v, T)]["status"] != "xotira yetmadi"
            for r in range(args.repeats)]
    random.Random(args.seed).shuffle(jobs)
    rows, all_keys = [], set()
    for idx, (v, T, r) in enumerate(jobs):
        # Tanaffus davomida tizimning umumiy protsessor yuklamasi o'lchanadi:
        # fon jarayonlari vaqtga ta'sir qilganini keyin aniqlash uchun.
        sys_cpu = psutil.cpu_percent(interval=args.pause) if (psutil and args.pause > 0) else None
        if not psutil or args.pause <= 0:
            time.sleep(max(args.pause, 0))
        try:
            freq = round(psutil.cpu_freq().current) if psutil and psutil.cpu_freq() else None
        except Exception:
            freq = None
        rb0 = disk_bytes()
        pre_mb = private_mb()
        with MemSampler() as ms:
            t0 = time.perf_counter()
            w, tp = run_variant(ctx, v, T)
            total = time.perf_counter() - t0
        rb1 = disk_bytes()
        row = dict(variant=v, T=T, rep=r, order=idx, total=round(total, 5),
                   sys_cpu_pause=sys_cpu, cpu_mhz=freq,
                   peak_rss_mb=round(ms.peak / 2 ** 20, 1) if psutil else None,
                   peak_private_mb=round(ms.peak_private / 2 ** 20, 1) if psutil else None,
                   private_pre_mb=pre_mb,
                   read_mb=round((rb1[0] - rb0[0]) / 2 ** 20, 1) if rb0 and rb1 else None,
                   write_mb=round((rb1[1] - rb0[1]) / 2 ** 20, 1) if rb0 and rb1 else None,
                   w_hash=sha(w), Q_hash=sha(ctx.maps[v]["Q"]) if args.write == "all" else None)
        row.update({k: round(x, 5) for k, x in tp.items()})
        all_keys |= set(tp)
        rows.append(row)
        print(f"[{idx + 1}/{len(jobs)}] {v} T={T} rep={r}: {total:.3f} s", flush=True)

    fields = ["variant", "T", "rep", "order", "total", "sys_cpu_pause", "cpu_mhz",
              "peak_rss_mb", "peak_private_mb", "private_pre_mb", "read_mb", "write_mb", "w_hash", "Q_hash"] + sorted(all_keys)
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def ci_median(x, B=10000, seed=0):
        x = np.asarray(x)
        rng = np.random.default_rng(seed)
        meds = np.median(rng.choice(x, (B, x.size)), axis=1)
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    summary = []
    for v, T in configs:
        g = [r for r in rows if r["variant"] == v and r["T"] == T]
        s = dict(variant=v, T=T, check=check[(v, T)])
        if g:
            tot = [r["total"] for r in g]
            lo, hi = ci_median(tot)
            s.update(runs=len(g), median_total=float(np.median(tot)), ci95=[lo, hi],
                     iqr=[float(np.percentile(tot, 25)), float(np.percentile(tot, 75))],
                     median_passes={k: float(np.median([r[k] for r in g])) for k in sorted(all_keys) if k in g[0]},
                     max_peak_rss_mb=max((r["peak_rss_mb"] or 0) for r in g),
                     median_read_mb=float(np.median([r["read_mb"] or 0 for r in g])),
                     median_write_mb=float(np.median([r["write_mb"] or 0 for r in g])),
                     max_peak_private_mb=max((r["peak_private_mb"] or 0) for r in g),
                     max_private_increase_mb=max(((r["peak_private_mb"] or 0) - (r["private_pre_mb"] or 0)) for r in g),
                     distinct_w=len({r["w_hash"] for r in g}),
                     distinct_Q=len({r["Q_hash"] for r in g}) if args.write == "all" else None)
        summary.append(s)
    with open(os.path.abspath(__file__), "rb") as fh:
        script_sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    env = dict(script=os.path.basename(__file__), script_sha256=script_sha, argv=sys.argv[1:],
               python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
               platform=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
               data=os.path.abspath(args.data), n=n, K=K, blocks=len(ctx.blk), block=args.block,
               alpha=args.alpha, nodata=None if np.isnan(args.nodata) else args.nodata,
               criteria=dict(zip(names, args.dirs)), seed=args.seed, repeats=args.repeats,
               pause=args.pause, write=args.write, w_ref=w_ref.tolist(),
               private_after_import_mb=base_import_mb, private_after_jit_mb=base_after_jit_mb,
               no_check=args.no_check,
               ram_total_gb=round(psutil.virtual_memory().total / 2 ** 30, 2) if psutil else None)
    json.dump(dict(env=env, summary=summary), open(os.path.join(args.out, "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nNatija (median, s):")
    for s in summary:
        if "median_total" in s:
            print(f"  {s['variant']:3} T={s['T']:2}: {s['median_total']:.3f}  CI95 [{s['ci95'][0]:.3f}; "
                  f"{s['ci95'][1]:.3f}]  tekshiruv: {s['check']['status']}")
        else:
            print(f"  {s['variant']:3} T={s['T']:2}: {s['check']['status']}")


if __name__ == "__main__":
    main()
