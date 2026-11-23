"""Step 9: draws the two figures of Chapter 5 (fig_model_comparison_human_f1.png and fig_feature_groups_human_f1.png) from the mean and standard deviation values typed in below, which are the values of Tables B.2 and B.3."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# ---------------- Figure 1: human-class F1 by split and model (mean +- sd over 25 folds) ----------------
splits = ["random", "grouped 0.9", "grouped 0.7\n(main)", "grouped 0.5"]
models = {  # name: (colour, marker, means, sds)
    "TF-IDF + LR (all words)":           ("#2a78d6", "o", [0.982, 0.978, 0.978, 0.961], [0.007, 0.011, 0.012, 0.031]),
    "Stylometric, random forest":        ("#eb6834", "s", [0.959, 0.948, 0.951, 0.933], [0.016, 0.018, 0.020, 0.026]),
    "TF-IDF + LR (function words only)": ("#1baf7a", "^", [0.908, 0.905, 0.892, 0.874], [0.020, 0.027, 0.026, 0.031]),
    "Stylometric, logistic regression":  ("#eda100", "D", [0.894, 0.892, 0.892, 0.887], [0.025, 0.020, 0.025, 0.028]),
}
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
offsets = [-0.15, -0.05, 0.05, 0.15]
for (name, (col, mk, mean, sd)), off in zip(models.items(), offsets):
    xs = [i + off for i in range(4)]
    ax.errorbar(xs, mean, yerr=sd, color=col, marker=mk, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1,
                linewidth=1.6, elinewidth=1.2, capsize=2.5, label=name)
for i, name in enumerate(models):                      # direct labels at the right end
    col, _, mean, _ = models[name]
    ax.annotate(f"{mean[-1]:.3f}", (3 + offsets[i], mean[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(4)); ax.set_xticklabels(splits)
ax.set_xlim(-0.4, 3.55); ax.set_ylim(0.82, 1.005)
ax.set_ylabel("F1 of the human class")
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Human-class F1 by type of test split (mean ± standard deviation over 25 test folds)", fontsize=10, loc="left", color=INK)
leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=2.4)
fig.tight_layout()
fig.savefig("fig_model_comparison_human_f1.png"); plt.close(fig)

# ---------------- Figure 2: feature groups, stylometric random forest, grouped 0.7 ----------------
rows = [  # label, mean, sd
    ("all 14 features", 0.951, 0.020, "all"),
    ("only punctuation and capitals", 0.862, 0.030, "only"),
    ("only voice", 0.782, 0.038, "only"),
    ("only length and rhythm", 0.662, 0.034, "only"),
    ("only word choice", 0.520, 0.058, "only"),
    ("without length and rhythm", 0.943, 0.020, "without"),
    ("without word choice", 0.936, 0.023, "without"),
    ("without voice", 0.927, 0.028, "without"),
    ("without the 2 capital-letter features", 0.900, 0.022, "without"),
    ("without punctuation and capitals", 0.863, 0.024, "without"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ys = []
y = 0
gaps = {"only": 0.7, "without": 0.7}
prev = None
for label, m, s, kind in rows:
    if prev is not None and kind != prev: y += gaps[kind]
    ys.append(y); y += 1; prev = kind
colour = {"all": "#2a78d6", "only": "#eb6834", "without": "#1baf7a"}
for (label, m, s, kind), yy in zip(rows, ys):
    ax.errorbar(m, yy, xerr=s, fmt="o", color=colour[kind], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1,
                elinewidth=1.4, capsize=2.5)
    ax.annotate(f"{m:.3f}", (m + s, yy), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.axvline(0.951, color="#2a78d6", linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.7)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0.45, 1.03)
ax.set_xlabel("F1 of the human class (grouped split, similarity ≥ 0.7)")
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.suptitle("Feature groups of the stylometric random forest (mean ± standard deviation, 25 folds)", fontsize=10, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("fig_feature_groups_human_f1.png"); plt.close(fig)
print("ok")