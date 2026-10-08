"""Replace only Tox21 FedAvg curves with the approved complete replay cohort."""

import json
from collections import defaultdict

import plot_approved_revision_20260920 as base
from publish_official_fedavg_reference_20260925 import REPLAY, load_replay


OUT = base.REVISION / "revision_approval_20260925/tox21_replay_figure"
FIGURES = base.ROOT / "paper_acs_latex/figures/revision_20260925"
STEM = "classification_convergence_official_tox21"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((base.OUTPUT / "figure_provenance.json").read_text())
    for source in provenance["convergence_sources"]:
        assert base.digest(base.Path(source["path"])) == source["sha256"]
    endpoints, _, replay_hashes, _ = load_replay()
    original = base.read_rows(base.OUTPUT / "convergence_seed_curves.csv")
    assert len(original) == 216 * 50
    plotted = []
    replay_rows = {}
    for alpha in base.ALPHAS:
        for seed in (0, 1, 2):
            case = f"Tox21_a{str(alpha).replace('.', 'p')}_seed{seed}_official_avg"
            path = REPLAY / "cases" / case / "masked_rounds.csv"
            replay_rows[alpha, seed] = (path, base.read_rows(path))
    for row in original:
        record = dict(row)
        record["protocol"] = "revised_paired"
        if row["dataset"] == "Tox21" and row["method"] == "avg":
            path, values = replay_rows[float(row["alpha"]), int(row["seed"])]
            record["server_test_auc"] = values[int(row["csv_round"])]["masked_test_auc"]
            record["source"] = str(path)
            record["protocol"] = "official_reference_different_split_and_supervision"
        plotted.append(record)
    assert sum(a["server_test_auc"] != b["server_test_auc"] for a, b in zip(original, plotted)) <= 450
    assert all(all(a[k] == b[k] for k in a) for a, b in zip(original, plotted)
               if not (a["dataset"] == "Tox21" and a["method"] == "avg"))
    per_seed = defaultdict(list)
    old_curves = defaultdict(list)
    for row in original:
        old_curves[row["dataset"], float(row["alpha"]), row["method"], int(row["seed"])].append(float(row["server_test_auc"]))
    for row in plotted:
        per_seed[row["dataset"], float(row["alpha"]), row["method"], int(row["seed"])].append(float(row["server_test_auc"]))
    assert len(per_seed) == 216 and all(len(v) == 50 for v in per_seed.values())
    curves, summary = {}, []
    for dataset in base.DATASETS:
        for alpha in base.ALPHAS:
            for method, *_ in base.METHODS:
                array = base.np.asarray([per_seed[dataset, alpha, method, s] for s in (0, 1, 2)])
                assert base.np.isfinite(array).all()
                assert ((array >= 0) & (array <= 1)).all()
                curves[dataset, alpha, method] = array.mean(0)
                protocol = ("official_reference_different_split_and_supervision"
                            if dataset == "Tox21" and method == "avg" else "revised_paired")
                for r in range(50):
                    summary.append(dict(dataset=dataset, alpha=alpha, method=method,
                        communication_round=r + 1, n_seeds=3, mean_auc=float(array[:, r].mean()),
                        sample_sd=float(array[:, r].std(ddof=1)), protocol=protocol))
                if dataset == "Tox21" and method == "avg":
                    assert all(abs(array[s, -1] - endpoints[dataset, str(alpha), s]) < 1e-12 for s in (0, 1, 2))

    base.plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": .75, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = base.plt.subplots(3, 3, figsize=(9.3, 6.45))
    handles = []
    for r, dataset in enumerate(base.DATASETS):
        old_means = [base.np.asarray([old_curves[dataset, a, m, s] for s in (0, 1, 2)]).mean(0)
                     for a in base.ALPHAS for m, *_ in base.METHODS]
        low = min(.5, base.np.floor((min(v.min() for v in old_means) - .015) * 20) / 20)
        high = min(1., base.np.ceil((max(v.max() for v in old_means) + .015) * 20) / 20)
        for c, alpha in enumerate(base.ALPHAS):
            ax = axes[r, c]
            for method, label, color, style in base.METHODS:
                mean = curves[dataset, alpha, method]
                assert mean.min() >= low and mean.max() <= high
                line, = ax.plot(range(1, 51), mean, color=color, linestyle=style,
                                linewidth=1.65 if method == "fedavg_proto" else 1.15)
                if r == c == 0:
                    handles.append(line)
            title = f"{dataset}, $\\alpha={alpha:.1f}$"
            if dataset == "Tox21":
                title += r" $\dagger$"
            ax.set_title(title, fontsize=10.2, pad=5)
            ax.set(xlim=(1, 50), ylim=(low, high), xticks=(1, 10, 20, 30, 40, 50))
            ax.set_xlabel("Communication round", labelpad=3)
            ax.set_ylabel("ROC-AUC", labelpad=3)
            ax.grid(color="#dedede", linewidth=.55, alpha=.8)
            ax.set_axisbelow(True)
    labels = ["FedAvg / FedAvg" + r"$^{\dagger}$"] + [x[1] for x in base.METHODS[1:]]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(.52, .040), handlelength=2.5, columnspacing=1.1, fontsize=9.2)
    fig.text(.5, .019, r"$\dagger$ Tox21 FedAvg: official-source reference with different splits and supervision; not a matched control.",
             ha="center", fontsize=8, color="#424242")
    fig.subplots_adjust(left=.074, right=.985, bottom=.18, top=.953, hspace=.55, wspace=.30)
    fig.canvas.draw()
    assert all(len(ax.lines) == 8 for ax in axes.flat)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{STEM}.{suffix}", bbox_inches="tight")
    fig.savefig(FIGURES / f"{STEM}.png", dpi=300, bbox_inches="tight")
    base.plt.close(fig)
    base.write_rows(OUT / "convergence_seed_curves.csv", plotted)
    base.write_rows(OUT / "convergence_mean_sd.csv", summary)
    (OUT / "VERIFICATION.json").write_text(json.dumps(dict(
        official_replay_source_hashes=replay_hashes, unchanged_other_curves=207,
        replaced_curves=9, communication_rounds=50, seeds=[0, 1, 2],
        endpoint_matches_replay=True, smoothing=False, error_bands=False,
        axes_ranges_preserved=True, mixed_protocol_reference_marked=True,
        original_figure_retained=True, source_data_modified=False,
        generated={str(p.relative_to(base.ROOT)): base.digest(p) for p in FIGURES.glob(STEM + ".*")}
    ), indent=2) + "\n")
    print("Verified 9 replay curves and 207 unchanged curves; all plotted endpoints match their source cohort.")
    print(FIGURES / (STEM + ".png"))


if __name__ == "__main__":
    main()
