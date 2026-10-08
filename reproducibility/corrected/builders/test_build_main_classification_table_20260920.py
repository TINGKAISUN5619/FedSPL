"""Small, training-free tests for the approved classification table builder."""

import copy
import unittest

import build_main_classification_table_20260920 as table


class TableTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(dataset="BBBP", seed="0", method="avg", descriptor="",
                             final_val="0.7", final_test="0.8")
        self.curve = [dict(round=str(i), scope="global", metric="auc", encoder="mpnn",
            partition_method="hetero", dataset="BBBP", method="avg", descriptor="",
            model_seed="0", split_seed="0", partition_seed="0", val="0.7", test="0.8",
            lambda_proto="", local_clusters="", global_clusters="", best_test="0.99") for i in range(50)]

    def test_final_not_best(self):
        self.assertEqual(table.validate_curve(self.curve, self.expected)["test"], "0.8")

    def test_missing_or_duplicate_round_rejected(self):
        with self.assertRaises(ValueError):
            table.validate_curve(self.curve[:-1], self.expected)
        self.curve[-1]["round"] = "48"
        with self.assertRaises(ValueError):
            table.validate_curve(self.curve, self.expected)

    def test_seed_and_scope_mismatch_rejected(self):
        for field, value in (("split_seed", "1"), ("scope", "clients_avg"), ("test", "nan")):
            changed = copy.deepcopy(self.curve)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                table.validate_curve(changed, self.expected)

    def test_manifest_grid_is_exact(self):
        with self.assertRaises(ValueError):
            table.validate_manifest([])
        self.assertEqual(len(table.expected_keys()), 360)

    def test_rank_before_rounding_and_exact_ties(self):
        self.assertEqual(table.dense_ranks({"a": 0.800041, "b": 0.800042, "c": 0.800041}),
                         {"a": 2, "b": 1, "c": 2})

    def test_pairs_match_seed_not_insertion_order(self):
        self.assertEqual(table.paired_values({2: 8, 0: 4, 1: 6}, {1: 4, 2: 5, 0: 3}), [1, 2, 3])
        with self.assertRaises(ValueError):
            table.paired_values({0: 1, 1: 2}, {0: 1, 1: 2})

    def test_sample_sd_and_paired_ci(self):
        mean, sd, low, high = table.paired_statistics([1, 2, 3])
        self.assertEqual((mean, sd), (2, 1))
        self.assertAlmostEqual(high-mean, table.T_CRITICAL/(3**0.5))
        self.assertAlmostEqual(mean-low, high-mean)

    def test_adapted_baseline_headers_and_disclosure(self):
        summary = [dict(dataset=d, alpha=a, method=m, mean_auc=0.7, sample_sd_auc=0.01, rank=3)
                   for d in table.DATASETS for a in table.ALPHAS for m in table.METHODS]
        central = [dict(dataset=d, mean_auc=0.9, sample_sd_auc=0.02) for d in table.DATASETS]
        tex = table.latex_table(summary, central)
        for name in (r"Proto\\adapted", r"FPL\\inspired", r"TGP\\inspired"):
            self.assertIn(name, tex)
        self.assertNotIn(r"$^\dagger$", tex)
        self.assertIn("not unmodified reference implementations", tex)

    def test_central_final_not_validation_selected(self):
        endpoints = [dict(checkpoint="final", split="test", block=49, auc=0.8, valid_tasks=1),
                     dict(checkpoint="val_selected", split="test", block=2, auc=0.99, valid_tasks=1)]
        self.assertEqual(table.central_final_endpoint(endpoints, "test")["auc"], 0.8)
        with self.assertRaises(ValueError):
            table.central_final_endpoint(endpoints[1:], "test")
        for field, value in (("block", 48), ("auc", float("nan")), ("valid_tasks", 0)):
            changed = copy.deepcopy(endpoints)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                table.central_final_endpoint(changed, "test")

    def test_central_once_per_dataset_and_not_ranked(self):
        summary = [dict(dataset=d, alpha=a, method=m, mean_auc=0.7, sample_sd_auc=0.01, rank=3)
                   for d in table.DATASETS for a in table.ALPHAS for m in table.METHODS]
        central = [dict(dataset=d, mean_auc=0.9, sample_sd_auc=0.02) for d in table.DATASETS]
        tex = table.latex_table(summary, central)
        self.assertEqual(tex.count(r"\multirow{3}{*}"), 5)
        self.assertEqual(tex.count("0.9000"), 5)
        self.assertNotIn(r"\textbf{0.9000}", tex)
        self.assertNotIn(r"\underline{0.9000}", tex)
        self.assertNotIn("pending", tex)
        with self.assertRaises(ValueError):
            table.latex_table(summary, central[:-1])

    def test_completed_central_sources(self):
        sources, summary = table.build_central()
        self.assertEqual(len(sources), 15)
        self.assertEqual(len(summary), 5)
        for row in summary:
            values = [r["final_test_auc"] for r in sources if r["dataset"] == row["dataset"]]
            self.assertEqual(row["mean_auc"], table.statistics.mean(values))
            self.assertEqual(row["sample_sd_auc"], table.statistics.stdev(values))
            self.assertEqual(row["alpha"], "not_applicable")
            self.assertFalse(row["ranked"])


if __name__ == "__main__":
    unittest.main()
