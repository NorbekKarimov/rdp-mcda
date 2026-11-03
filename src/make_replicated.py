#!/usr/bin/env python3
"""
Xotiradan katta sinov uchun ma'lumot: real to'plam k marta ketma-ket ulanadi.

Kirish : (n, K) float32 .npy (masalan, data\\orol_v2_30m\\data.npy) va uning meta.json fayli.
Chiqish: (n, k*K) float32 .npy; har bir mezon qatori asl qatorning k ta nusxasidan iborat.
         meta.json nusxalanadi va unga "replicated_from", "replication_factor" qo'shiladi.

Qiymatlar o'zgartirilmaydi: fayl faqat hisoblash xatti-harakatini (vaqt, xotira,
xotiradan tashqari ishlash) sinash uchun mo'ljallangan.

Misol:
  python make_replicated.py --src data\\orol_v2_30m\\data.npy --k 4 --dst data\\orol_v2_30m_x4\\data.npy
"""
import argparse, json, os, shutil, sys, time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1 << 24, help="bir martada ko'chiriladigan kataklar")
    args = ap.parse_args()

    X = np.load(args.src, mmap_mode="r")
    n, K = X.shape
    need = n * K * args.k * 4
    d = os.path.dirname(os.path.abspath(args.dst))
    os.makedirs(d, exist_ok=True)
    free = shutil.disk_usage(d).free
    print(f"manba: {X.shape} {X.dtype}; natija: ({n}, {args.k * K}), {need / 2**30:.2f} GiB; "
          f"bo'sh joy: {free / 2**30:.1f} GiB", flush=True)
    if free < need * 1.05:
        sys.exit("Diskda joy yetmaydi")

    Y = np.lib.format.open_memmap(args.dst, mode="w+", dtype=np.float32, shape=(n, args.k * K))
    t0 = time.perf_counter()
    for j in range(n):
        for c in range(args.k):
            off = c * K
            for s in range(0, K, args.chunk):
                e = min(s + args.chunk, K)
                Y[j, off + s:off + e] = X[j, s:e]
        Y.flush()
        print(f"  mezon {j + 1}/{n} tayyor ({time.perf_counter() - t0:.0f} s)", flush=True)
    del Y

    src_meta = os.path.join(os.path.dirname(os.path.abspath(args.src)), "meta.json")
    meta = json.load(open(src_meta, encoding="utf-8")) if os.path.exists(src_meta) else {}
    meta.update(replicated_from=os.path.abspath(args.src), replication_factor=args.k,
                note="Real ma'lumot k marta ketma-ket ulangan; faqat hisoblash sinovi uchun. "
                     "Geofazoviy atributlar (transform, width) bu faylga mos emas.")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"tayyor: {args.dst} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
