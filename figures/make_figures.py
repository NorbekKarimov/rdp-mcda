import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
INK, QUIET, EDGE, GRID = "#1f1f1f", "#5f5f5f", "#9a9a92", "#e4e4df"
MUTED, ACC = "#b9b9b0", "#2a74d5"
PT = 0.72  # svg px -> pt


def canvas(W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, dash=False, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=(ACC if accent else "none"), alpha=None,
                                ec=(ACC if accent else EDGE), lw=(2 if accent else 1.2),
                                ls=("--" if dash else "-")))
    if accent:
        ax.patches[-1].set_facecolor((42 / 255, 116 / 255, 213 / 255, 0.12))


def txt(ax, x, y, s, size=13, bold=False, color=INK, ha="left"):
    ax.text(x, y, s, fontsize=size * PT, fontweight=("bold" if bold else "normal"), color=color,
            ha=ha, va="baseline")


def arrow(ax, pts):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=EDGE, lw=1.2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2, mutation_scale=10))


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


# ---------------- 1-rasm: umumiy sxema
fig, ax = canvas(760, 756)
txt(ax, 24, 30, "Four full passes and one light pass compute the whole chain", 15, True)
x0, W, cx = 160, 440, 380
blocks = [
    (52, 56, False, "Input: raster stack X", ["n × K, float32, memory-mapped .npy file"], None),
    (148, 56, False, "Pass 1 · statistics", ["per block: valid-cell count, min_j, max_j"], 1),
    (244, 72, False, "Pass 2 · histogram", ["per block: integer histogram with N_b bins",
                                                 "quantiles give L_j and U_j (Lemma 1)"], 1),
    (356, 72, False, "Pass 3 · EWM weights in one pass", ["accumulate R_j = Σ r_ij and Q_j = Σ r_ij ln r_ij",
                                                                "w_j from Equation (2) at the end of the pass"], 1),
    (468, 88, False, "Pass 4 · SAW, TOPSIS and VIKOR scores", ["S_i and C_i with fixed ideal points (Lemma 2)",
                                                                   "S^V_i and R_i written to disk",
                                                                   "S*, S⁻, R*, R⁻ found in the same pass"], 1),
    (596, 56, True, "Light pass · VIKOR Q_i, Eq. (3)", ["reads only S^V_i and R_i: 2/n of the data"], 1),
    (692, 40, False, "Output: weights w_j and maps S, C, Q", [], None),
]
for y, h, dash, name, lines, par in blocks:
    box(ax, x0, y, W, h, dash)
    txt(ax, 176, y + 24 if lines else y + 25, name, 13, True)
    for k, l in enumerate(lines):
        txt(ax, 176, y + 40 + 16 * k, l, 11.5, color=QUIET)
    if par:
        txt(ax, 584, y + 24, "parallel, T workers", 11.5, color=ACC, ha="right")
edges = [(108, 148, "P = ⌈K/b⌉ blocks"), (204, 244, "m, min_j, max_j (merged by one thread)"),
         (316, 356, "L_j, U_j (computed by one thread)"), (428, 468, "w_j (merged by one thread)"),
         (556, 596, "S*, S⁻, R*, R⁻ (merged by one thread)"), (652, 692, "")]
for a, b, lab in edges:
    arrow(ax, [(cx, a), (cx, b)])
    if lab:
        txt(ax, 392, (a + b) / 2 + 4, lab, 11.5, color=QUIET)
save(fig, "fig1")

# ---------------- 2-rasm: bitta o'tish ichida
fig, ax = canvas(760, 370)
txt(ax, 24, 30, "Blocks run in any order; partial results are merged in a fixed order", 15, True)
box(ax, 24, 70, 160, 140)
txt(ax, 40, 94, "Block queue", 13, True)
for k, l in enumerate(["k = 1, 2, …, P", "an idle worker takes", "the next block"]):
    txt(ax, 40, 112 + 16 * k, l, 11.5, color=QUIET)
for k, (y, nm) in enumerate([(70, "Worker 1"), (120, "Worker 2"), (170, "Worker T")]):
    box(ax, 240, y, 180, 40)
    txt(ax, 256, y + 25, nm)
    arrow(ax, [(184, y + 20), (240, y + 20)])
    arrow(ax, [(420, y + 20), (456, y + 20)])
box(ax, 456, 70, 280, 140)
txt(ax, 472, 94, "Partial results Π[1], …, Π[P]", 13, True)
txt(ax, 472, 112, "block k writes only to Π[k]", 11.5, color=QUIET)
txt(ax, 472, 128, "write order does not matter", 11.5, color=QUIET)
box(ax, 24, 270, 160, 76)
txt(ax, 40, 294, "Map slices", 13, True)
txt(ax, 40, 312, "written in place", 11.5, color=QUIET)
txt(ax, 40, 328, "for block k", 11.5, color=QUIET)
box(ax, 240, 270, 180, 60)
txt(ax, 256, 294, "Global result", 13, True)
txt(ax, 256, 312, "to the next pass", 11.5, color=QUIET)
box(ax, 456, 270, 280, 76, accent=True)
txt(ax, 472, 294, "Merge: one thread", 13, True)
txt(ax, 472, 312, "Π[1] ⊕ Π[2] ⊕ … ⊕ Π[P], fixed order", 11.5)
txt(ax, 472, 328, "result independent of T (Theorem 2)", 11.5)
arrow(ax, [(596, 210), (596, 270)])
arrow(ax, [(456, 300), (420, 300)])
arrow(ax, [(330, 210), (330, 240), (104, 240), (104, 270)])
save(fig, "fig2")

# ---------------- 3-rasm: vaqt diagrammasi
fig, ax = canvas(760, 316)
txt(ax, 24, 30, "Workers take blocks from a queue; the merge runs on one thread after the barrier", 15, True)
for y, nm in [(91, "Worker 1"), (131, "Worker 2"), (171, "Worker T"), (219, "one thread")]:
    txt(ax, 24, y, nm)
rows = [(72, [(130, 100, "1"), (234, 86, "4"), (324, 96, "7")]),
        (112, [(130, 120, "2"), (254, 86, "5"), (344, 66, "8")]),
        (152, [(130, 80, "3"), (214, 116, "6"), (334, 92, "P")])]
for y, bars in rows:
    for x, w, l in bars:
        ax.add_patch(FancyBboxPatch((x, y), w, 28, boxstyle="round,pad=0,rounding_size=6", fc="none", ec=EDGE, lw=1.2))
        txt(ax, x + w / 2, y + 18, f"Block {l}", 11.5, ha="center")
    ax.add_patch(FancyBboxPatch((576, y), 104, 28, boxstyle="round,pad=0,rounding_size=6", fc="none", ec=EDGE, lw=1.2, ls="--"))
txt(ax, 628, 90, "next pass", 11.5, color=QUIET, ha="center")
ax.plot([440, 440], [64, 236], color=EDGE, lw=1.2, ls="--")
txt(ax, 440, 56, "barrier", 11.5, color=QUIET, ha="center")
box(ax, 448, 200, 112, 28, accent=True)
txt(ax, 504, 218, "Π[1] ⊕ … ⊕ Π[P]", 11.5, ha="center")
ax.plot([130, 130, 436, 436], [246, 252, 252, 246], color=EDGE, lw=1.2)
ax.plot([448, 448, 560, 560], [246, 252, 252, 246], color=EDGE, lw=1.2)
txt(ax, 283, 270, "parallel part: max(t_comp / min(T, P), V / B(T))", 11.5, ha="center")
txt(ax, 504, 270, "serial part t_s", 11.5, ha="center")
arrow(ax, [(130, 292), (736, 292)])
txt(ax, 736, 284, "time", 11.5, color=QUIET, ha="right")
save(fig, "fig3")


def f(v, d=2):
    return f"{v:.{d}f}"


# ---------------- 4-rasm: ablyatsiya
rows = [("B1 · 9 passes, NumPy", 96.477), ("A1 · 4 + 1, NumPy", 44.715), ("A2 · 9 passes, Numba", 11.796),
        ("P · 4 + 1, Numba, T = 1", 6.857), ("P · 4 + 1, Numba, T = 12", 1.842)]
fig, ax = canvas(760, 310)
txt(ax, 24, 30, "30 m stack: from 96.5 s to 1.84 s, 52 times faster", 15, True)
txt(ax, 24, 52, "Median time; right: speedup relative to B1", 12, color=QUIET)
base = rows[0][1]
for i, (lab, t) in enumerate(rows):
    y = 80 + i * 44
    w = max(t / base * 380, 2)
    txt(ax, 24, y + 18, lab, 12)
    ax.add_patch(Rectangle((230, y), w, 26, fc=(ACC if i == 4 else MUTED), ec="none"))
    txt(ax, 230 + w + 8, y + 18, f"{f(t, 2 if t < 10 else 1)} s · {f(base / t, 1)}×", 12)
save(fig, "fig4")

# ---------------- 5-rasm: tezlanish
T = [1, 2, 4, 6, 8, 12]
P = [6.857, 3.661, 2.225, 1.891, 1.868, 1.842]
P5 = [7.625, 3.941, 2.256, 1.894, 1.840, 1.774]
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
sp = [P[0] / v for v in P]
sp5 = [P5[0] / v for v in P5]
ax.plot([1, 8], [1, 8], ls="--", color=EDGE, lw=1)
ax.text(8.2, 7.8, "linear speedup", color=QUIET, fontsize=9)
ax.axvline(6, ls=":", color=EDGE, lw=1)
ax.text(6.15, 0.35, "6 physical cores", color=QUIET, fontsize=9)
ax.plot(T, sp5, "-o", color="#555555", ms=4, lw=2)
ax.plot(T, sp, "-o", color=ACC, ms=4, lw=2)
ax.text(12.2, sp5[-1], f"P5 {f(sp5[-1])}×", color="#555555", fontsize=9, va="center")
ax.text(12.2, sp[-1] - 0.25, f"P {f(sp[-1])}×", color=ACC, fontsize=9, va="center")
ax.set_xticks(T)
ax.set_xlim(0.6, 13.6)
ax.set_ylim(0, 8.3)
ax.set_xlabel("number of threads T", color=QUIET, fontsize=9)
ax.set_ylabel("speedup, ×", color=QUIET, fontsize=9)
ax.grid(axis="y", color=GRID)
ax.tick_params(labelsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("Speedup levels off after six cores: 3.72× for P and 4.30× for P5 at T = 12", loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/fig5.png", dpi=200, facecolor="white")
plt.close(fig)

# ---------------- 6-rasm: o'tishlar ulushi
parts = ["statistics", "histogram", "EWM", "scores", "Q (light / 5th pass)", "flush"]
data = [("P, T = 1", [0.626, 1.215, 2.939, 1.681, 0.160, 0.233]),
        ("P, T = 12", [0.152, 0.302, 0.522, 0.493, 0.128, 0.238]),
        ("P5, T = 1", [0.621, 1.223, 2.938, 1.659, 1.184, 0]),
        ("P5, T = 12", [0.151, 0.305, 0.519, 0.551, 0.244, 0])]
fig, ax = canvas(760, 330)
txt(ax, 24, 30, "At 12 threads the serial flush takes 13% of P's time; P5 has no flush", 15, True)
txt(ax, 24, 52, "Share of each pass, %; right: sum of pass times; 30 m, final series", 12, color=QUIET)
greys = ["#e3e3de", "#cfcfc8", "#bcbcb3", "#a5a59b", "#8c8c82", ACC]
for i, (lab, vals) in enumerate(data):
    y = 80 + i * 52
    tot = sum(vals)
    txt(ax, 24, y + 20, lab, 12)
    acc = 0
    for k, v in enumerate(vals):
        if v <= 0:
            continue
        x = 120 + acc / tot * 540
        w = v / tot * 540
        ax.add_patch(Rectangle((x, y), max(w - 1, 1), 30, fc=greys[k], ec="none"))
        if v / tot >= 0.08:
            txt(ax, x + w / 2, y + 20, f"{round(v / tot * 100)}%", 12, ha="center")
        acc += v
    txt(ax, 668, y + 20, f"{f(tot)} s", 12)
lx = 24
for k, p in enumerate(parts):
    ax.add_patch(Rectangle((lx, 296), 12, 12, fc=greys[k], ec="none"))
    txt(ax, lx + 18, 306, p, 12, color=QUIET)
    lx += len(p) * 6.4 + 34
save(fig, "fig6")

# ---------------- 7-rasm: xotira log-log
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
pc = [4149251, 46060601, 184242404]
pm = [21.2, 20.2, 20.1]
ax.plot(pc, pm, "-o", color=ACC, lw=2, ms=5)
for c, m in zip(pc, pm):
    ax.text(c, m * 0.62, f"{f(m, 1)} MB", color=ACC, ha="center", fontsize=9)
ax.plot([4149251, 46060601], [645.5, 7164], ls="--", color="#555555", lw=1.5)
ax.plot([4149251], [645.5], "o", color="#555555", ms=6)
ax.plot([46060601], [7164], "o", mfc="white", mec="#555555", ms=6)
ax.text(4149251, 645.5 * 0.55, "B0: 645.5 MB", color="#555555", ha="center", fontsize=9)
ax.text(52000000, 7164 * 0.5, "B0 estimate: ≈ 7164 MB", color="#555555", fontsize=9)
ax.axhline(7588, ls="--", color=EDGE, lw=1)
ax.text(1.1e6, 7588 * 1.25, "main memory: 7588 MB (7.41 GiB)", color=QUIET, fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e6, 1e9)
ax.set_ylim(1, 2e4)
ax.set_xticks([1e6, 1e7, 1e8, 1e9])
ax.set_xticklabels(["1 M", "10 M", "100 M", "1 B"])
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(["1", "10", "100", "1 000", "10 000"])
ax.set_xlabel("valid cells (log)", color=QUIET, fontsize=9)
ax.set_ylabel("memory growth, MB (log)", color=QUIET, fontsize=9)
ax.grid(axis="y", color=GRID, which="major")
ax.tick_params(labelsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("Memory growth: P stays near 20 MB, B0 grows with the data", loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/fig7.png", dpi=200, facecolor="white")
plt.close(fig)

# ---------------- 8-rasm: x4 o'tishlari
bars = [("A2 (9 passes)", 47.45, 57.7, [("1", 5.490), ("2", 5.481), ("3", 5.391), ("4", 5.971), ("5", 5.248),
                                         ("6", 4.965), ("7", 5.160), ("8", 4.685), ("9", 4.820)]),
        ("P (4 + light)", 27.99, 27.6, [("1", 5.012), ("2", 4.996), ("3", 5.430), ("4*", 9.401), ("Q", 0.666), ("fl.", 0.738)]),
        ("P5 (5 passes)", 28.48, 27.9, [("1", 5.368), ("2", 5.871), ("3", 5.553), ("4", 4.654), ("5", 5.134)])]
fig, ax = canvas(760, 320)
txt(ax, 24, 30, "P reads 2.09 times less, but its writing pass 4 takes 9.4 s", 15, True)
txt(ax, 24, 52, "9.73 GB stack, T = 12; pass times (median); right: total time and volume read from disk", 12, color=QUIET)
mx = max(b[1] for b in bars)
for i, (lab, t, rd, ps) in enumerate(bars):
    y = 84 + i * 66
    txt(ax, 24, y + 22, lab, 12)
    acc = 0
    for k, (p, v) in enumerate(ps):
        x = 150 + acc / mx * 470
        w = v / mx * 470
        col = ACC if p == "4*" else ("#cfcfc8" if k % 2 else "#e3e3de")
        ax.add_patch(Rectangle((x, y), max(w - 1, 1), 34, fc=col, ec="none"))
        if w >= 18:
            txt(ax, x + w / 2, y + 22, p, 12, ha="center")
        acc += v
    txt(ax, 628, y + 22, f"{f(t)} s · {f(rd, 1)} GiB", 12)
txt(ax, 24, 300, "4* — pass that writes S^V and R (2.30 GiB); Q — light pass; fl. — flush", 12, color=QUIET)
save(fig, "fig8")

# ---------------- 9-rasm: sinf o'zgarishlari
rows = [("A3, float64: order effect", 0), ("A3, float32: order effect", 31),
        ("P, float32: precision effect", 242), ("A3, float32: order + precision", 269)]
fig, ax = canvas(760, 270)
txt(ax, 24, 30, "Single precision alone changes 242 cells; summation order changes at most 31", 15, True)
txt(ax, 24, 52, "Cells changing class (of 46,060,601), maximum over 100 runs; SAW, 5 classes", 12, color=QUIET)
mx = max(v for _, v in rows)
for i, (lab, v) in enumerate(rows):
    y = 80 + i * 44
    w = max(v / mx * 400, 1.5)
    txt(ax, 24, y + 18, lab, 12)
    ax.add_patch(Rectangle((250, y), w, 26, fc=(ACC if i == 2 else MUTED), ec="none"))
    txt(ax, 250 + w + 8, y + 18, str(v), 12)
save(fig, "fig9")
print("ok")
