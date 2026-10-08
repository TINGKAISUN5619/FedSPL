#!/usr/bin/env python3
"""Local, standard-library-only R1.5 audit; never import or execute training code."""

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/reviewer_revision_20260831"
SOURCE = BASE / "server_synced/formal_v3"
OUTPUT = BASE / "cluster_support_review_20260916"
KS = (8, 16, 32, 64, 128)
SEEDS = (0, 1, 2)
ROUNDS = 50
COUNT_FIELDS = ("realized_k", "samples", "singleton_clusters", "clusters_lt3", "clusters_lt5")
SHAPE_FIELDS = COUNT_FIELDS + ("min_count", "p25_count", "median_count", "mean_count", "max_count")
VARIABLE_CONFIG = {"seed", "model_seed", "split_seed", "partition_seed", "results_dir",
                   "topoproto_clusters", "topoproto_global_clusters"}
FIXED_CONFIG = dict(dataset="SIDER", fedmid="fedavg_proto", part_alpha=0.5,
                    partition_method="hetero", encoder="mpnn", proto_descriptor_space="fingerprint",
                    lambda_proto=0.1, topoproto_match="hard", comm_round=50,
                    local_steps_per_round=200, numClient=4, clients_per_round=4,
                    global_only_eval=True, eval_frequency=1, batch_size=64)
OCCUPANCY_FIELDS = (
    "time,dataset,round,scope,client_idx,requested_k,realized_k,samples,min_count,"
    "p25_count,median_count,mean_count,max_count,singleton_clusters,clusters_lt3,clusters_lt5,"
    "cluster_cache_hit,cluster_fit_seconds,collection_seconds,model_seed,split_seed,"
    "partition_seed,encoder,descriptor"
).split(",")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(items, context):
    items = list(items)
    require(len(items) == 1, f"Expected exactly one {context}; found {len(items)}")
    return items[0]


def number(value):
    value = float(value)
    require(math.isfinite(value), "Nonfinite number")
    return value


def integer(value):
    value = number(value)
    require(value.is_integer(), f"Noninteger count: {value}")
    return int(value)


def rel(path):
    return str(path.resolve().relative_to(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path, inventory, role):
    remember(path, inventory, role)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    require(fields and rows, f"Empty CSV: {path}")
    require(len(fields) == len(set(fields)), f"Duplicate CSV fields: {path}")
    require(all(None not in r and None not in r.values() for r in rows), f"Malformed CSV: {path}")
    return fields, rows


def remember(path, inventory, role):
    inventory[rel(path)] = dict(path=rel(path), role=role, bytes=path.stat().st_size,
                                sha256=digest(path))


def indexed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, f"Duplicate key: {key}")
        result[key] = row
    return result


def final_metric(rows):
    require(all(r["scope"] == "global" for r in rows), "Unexpected metric scope")
    index = indexed(rows, ("round",))
    require(set(index) == {(str(i),) for i in range(ROUNDS)}, "Missing/extra metric rounds")
    return index[(str(ROUNDS - 1),)]


def common_config(config):
    return {k: v for k, v in config.items() if k not in VARIABLE_CONFIG}


def check_config(config, k, seed):
    expected = dict(FIXED_CONFIG, topoproto_clusters=k, topoproto_global_clusters=k,
                    seed=seed, model_seed=seed, split_seed=seed, partition_seed=seed)
    for key, value in expected.items():
        require(config.get(key) == value, f"Protocol mismatch: {key}={config.get(key)!r}, expected {value!r}")


def support(rows):
    # Threshold numerators are logged directly, never inferred from a mean/quantile.
    values = {key: sum(integer(r[key]) for r in rows) for key in COUNT_FIELDS}
    for row in rows:
        n, samples, single, lt3, lt5 = (integer(row[key]) for key in COUNT_FIELDS)
        require(0 <= single <= lt3 <= lt5 <= n <= samples, "Invalid support counts")
        require(n <= integer(row["requested_k"]), "Realized K exceeds requested K")
        if n:
            require(number(row["min_count"]) >= 1, "Nonempty-cluster minimum is below one")
        else:
            require(samples == 0, "Samples present but no nonempty cluster")
    n = values.pop("realized_k")
    values["nonempty_clusters"] = n
    for field, name in (("singleton_clusters", "singleton_fraction"),
                        ("clusters_lt3", "lt3_fraction"), ("clusters_lt5", "lt5_fraction")):
        values[name] = values[field] / n if n else None
    return values


def parse_log(path, case, inventory):
    remember(path, inventory, "original_run_log")
    text = path.read_text(encoding="utf-8")
    configs = [(i, ast.literal_eval(line)) for i, line in enumerate(text.splitlines(), 1)
               if line.startswith("{'dataset':")]
    config_line, config = one(configs, "logged configuration")
    starts = re.findall(r"^\[([^\]]+)\] START index=(\d+) gpu=(\d+) suite=(\S+) case=(\S+)$", text, re.M)
    ends = re.findall(r"^\[([^\]]+)\] END index=(\d+) status=(\S+)$", text, re.M)
    start = one(starts, "START marker")
    end = one(ends, "END marker")
    require(start[4] == case and start[3] == "k_occupancy", "Wrong case in log")
    require(start[1] == end[1] and end[2] == "complete", "Incomplete or mismatched log")
    elapsed = (datetime.fromisoformat(end[0]) - datetime.fromisoformat(start[0])).total_seconds()
    require(elapsed > 0, "Nonpositive logged wall time")
    sizes = re.findall(r"client_idx = (\d+), local_sample_number = (\d+)", text)
    require(len(sizes) == 4 and {int(c) for c, _ in sizes} == set(range(4)), "Missing client sample sizes")
    rounds = {}
    current = None
    for line_no, line in enumerate(text.splitlines(), 1):
        match = re.search(r"fedavg_proto Communication round : (\d+)", line)
        if match:
            current = int(match[1])
            require(current not in rounds, "Duplicate communication round")
            rounds[current] = dict(source_line=line_no, clients=[], trained=[], collected=[], uploaded=[], aggregated=[])
        if current is None:
            continue
        r = rounds[current]
        if "client_indexes = " in line:
            r["clients"].append(ast.literal_eval(line.split("client_indexes = ", 1)[1]))
        match = re.search(r"\[Client (\d+)\] TopoFedProto train finished: round=(\d+)", line)
        if match:
            require(int(match[2]) == current, "Training/log round mismatch")
            r["trained"].append(int(match[1]))
        match = re.search(r"\[Client (\d+)\] Collected (\d+)/(\d+) local topology prototypes from (\d+) samples", line)
        if match:
            r["collected"].append(tuple(map(int, match.groups())))
        match = re.search(r"Server descriptor clustering: requested=(\d+), uploaded=(\d+), cache_hit=(True|False)", line)
        if match:
            r["uploaded"].append((int(match[1]), int(match[2]), match[3]))
        match = re.search(r"Round (\d+) aggregated (\d+) global topology prototypes", line)
        if match:
            require(int(match[1]) == current, "Aggregation/log round mismatch")
            r["aggregated"].append(int(match[2]))
    require(list(rounds) == list(range(ROUNDS)), "Missing/reordered communication rounds")
    endpoints = {}
    for split in ("Val", "Test"):
        values = re.findall(rf"^cur {split} Steps: (\d+) auc (\S+) std (\S+)$", text, re.M)
        require([int(v[0]) for v in values] == list(range(ROUNDS)), "Missing logged AUC rounds")
        endpoints[split.lower()] = {int(r): (number(v), number(sd)) for r, v, sd in values}
    return dict(config=config, config_line=config_line, start=start[0], end=end[0],
                case_index=int(start[1]), gpu=int(start[2]), wall_seconds=elapsed,
                client_sizes={int(c): int(n) for c, n in sizes}, rounds=rounds, endpoints=endpoints)


def audit_case(manifest, status, inventory):
    case = manifest["case_id"]
    k, seed = int(manifest["k"]), int(manifest["seed"])
    directory = SOURCE / "priority_suite/k_occupancy" / case
    occupancy = one(directory.glob("*_prototype_occupancy.csv"), f"occupancy CSV for {case}")
    metric = one((p for p in directory.glob("*.csv") if p != occupancy), f"metric CSV for {case}")
    require(occupancy.stem == metric.stem + "_prototype_occupancy", "CSV stem mismatch")
    require(set(directory.iterdir()) == {occupancy, metric}, "Unexpected case artifact; re-audit schema")
    log_path = one((SOURCE / "priority_suite/logs").glob(case + "_*.log"), f"run log for {case}")
    ofields, occ = read_csv(occupancy, inventory, "occupancy_aggregates")
    mfields, metrics = read_csv(metric, inventory, "round_auc")
    require(ofields == OCCUPANCY_FIELDS, "Occupancy schema changed; inspect before proceeding")
    log = parse_log(log_path, case, inventory)
    cfg = log["config"]
    check_config(cfg, k, seed)
    require(log["case_index"] == int(manifest["case_index"]), "Manifest/log index mismatch")
    require(Path(cfg["results_dir"]).parts[-3:] == ("priority_suite", "k_occupancy", case), "Logged result directory mismatch")
    require(Path(status["csv_file"]).parts[-4:] == ("priority_suite", "k_occupancy", case, metric.name), "Status/CSV mismatch")
    require(status["status"] == "complete" and status["target_round"] == "49" and status["max_round"] == "49", "Incomplete summary status")
    for field in ("suite", "case_id", "dataset", "alpha", "partition", "seed", "method", "encoder", "descriptor", "k", "lambda_proto"):
        require(status[field] == manifest[field], f"Manifest/status mismatch: {field}")
    require(int(manifest["rounds"]) == ROUNDS and int(manifest["local_steps"]) == 200, "Manifest budget mismatch")
    require(manifest["dataset"] == "SIDER" and number(manifest["alpha"]) == 0.5
            and manifest["partition"] == "hetero" and manifest["method"] == "fedavg_proto"
            and manifest["encoder"] == "mpnn" and manifest["descriptor"] == "fingerprint"
            and number(manifest["lambda_proto"]) == 0.1, "Wrong manifest protocol")
    final = final_metric(metrics)
    for split in ("val", "test"):
        require(math.isclose(number(final[split]), number(status["final_" + split]), abs_tol=1e-12, rel_tol=0), "Summary/final AUC mismatch")
    metadata = dict(dataset="SIDER", encoder="mpnn", descriptor="fingerprint",
                    model_seed=str(seed), split_seed=str(seed), partition_seed=str(seed))
    for row in occ + metrics:
        for field, value in metadata.items():
            require(row[field] == value, f"CSV metadata mismatch: {field}")
    for row in metrics:
        require(row["method"] == "fedavg_proto" and row["metric"] == "auc"
                and row["partition_method"] == "hetero" and number(row["lambda_proto"]) == 0.1
                and integer(row["local_clusters"]) == k and integer(row["global_clusters"]) == k,
                "Metric protocol mismatch")
        for split in ("val", "test"):
            value = number(row[split])
            require(0 <= value <= 1, "AUC outside [0,1]")
            require(math.isclose(value, log["endpoints"][split][int(row["round"])][0], abs_tol=1e-12, rel_tol=0), "Log/CSV AUC mismatch")
    scopes = [("client", str(c)) for c in range(4)] + [("server", "server")]
    oi = indexed(occ, ("round", "scope", "client_idx"))
    require(set(oi) == {(str(r), s, c) for r in range(ROUNDS) for s, c in scopes}, "Missing/extra occupancy rows")
    source_lines = {id(r): i for i, r in enumerate(occ, 2)}
    metric_lines = {r["round"]: i for i, r in enumerate(metrics, 2)}
    for row in occ:
        require(integer(row["requested_k"]) == k, "Occupancy K mismatch")
        support([row])
        require(row["cluster_cache_hit"] in ("True", "False"), "Unknown cache state")
        require(0 <= number(row["cluster_fit_seconds"]) <= number(row["collection_seconds"]), "Invalid timer values")
    stable = all(len({tuple(oi[str(r), s, c][f] for f in SHAPE_FIELDS) for r in range(ROUNDS)}) == 1 for s, c in scopes)
    round_rows = []
    for r in range(ROUNDS):
        clients = [oi[str(r), "client", str(c)] for c in range(4)]
        server = oi[str(r), "server", "server"]
        local, global_ = support(clients), support([server])
        require(local["samples"] == global_["samples"], "Local/server support total mismatch")
        for c, row in enumerate(clients):
            require(integer(row["samples"]) == log["client_sizes"][c], "Client sample-size mismatch")
        lr = log["rounds"][r]
        require(lr["clients"] and all(x == list(range(4)) for x in lr["clients"]), "Wrong participating clients")
        require(sorted(lr["trained"]) == list(range(4)), "Not all four clients trained exactly once")
        require(sorted(lr["collected"]) == [(c, integer(row["realized_k"]), k, integer(row["samples"])) for c, row in enumerate(clients)], "Log/local occupancy mismatch")
        requested, uploaded, cache_hit = one(lr["uploaded"], "upload record per round")
        require(requested == k and uploaded == local["nonempty_clusters"] and cache_hit == server["cluster_cache_hit"], "Upload record mismatch")
        require(one(lr["aggregated"], "aggregation per round") == global_["nonempty_clusters"], "Global cluster mismatch")
        record = dict(case_id=case, k=k, seed=seed, round=r, training_clients=4,
                      local_nonempty_clusters=local["nonempty_clusters"], server_nonempty_clusters=global_["nonempty_clusters"],
                      uploaded_prototype_records=uploaded,
                      local_cache_hits=sum(x["cluster_cache_hit"] == "True" for x in clients),
                      server_cache_hit=server["cluster_cache_hit"] == "True",
                      metric_source=rel(metric), metric_csv_line=metric_lines[str(r)],
                      occupancy_source=rel(occupancy), occupancy_csv_lines=";".join(str(source_lines[id(x)]) for x in clients + [server]),
                      log_source=rel(log_path), log_round_start_line=lr["source_line"])
        for scope, rows in (("local", clients), ("server", [server])):
            for field in ("cluster_fit_seconds", "collection_seconds"):
                record[scope + "_" + field] = sum(number(x[field]) for x in rows)
        round_rows.append(record)
    details = []
    for scope, client in scopes:
        for r in (0, ROUNDS - 1):
            row = oi[str(r), scope, client]
            details.append(dict(case_id=case, k=k, seed=seed, scope=scope, client_idx=client,
                                round=r, requested_k=k, **support([row]),
                                **{f: number(row[f]) for f in SHAPE_FIELDS if f not in COUNT_FIELDS},
                                occupancy_stats_unchanged_all_rounds=stable,
                                final_test_auc=number(final["test"]),
                                occupancy_source=rel(occupancy), occupancy_csv_line=source_lines[id(row)]))
    run = dict(case_id=case, k=k, seed=seed, model_seed=seed, split_seed=seed, partition_seed=seed,
               dataset="SIDER", suite="k_occupancy", alpha=0.5, partition="hetero", method="fedavg_proto",
               encoder="mpnn", descriptor="fingerprint", lambda_proto=0.1, local_steps_per_round=200,
               rounds_verified=ROUNDS, final_round=49, training_clients=4,
               client_sample_counts_json=json.dumps(log["client_sizes"], sort_keys=True),
               occupancy_stats_unchanged_all_rounds=stable,
               per_cluster_counts_available=False,
               final_val_auc=number(final["val"]), final_test_auc=number(final["test"]),
               final_val_task_sd_logged=log["endpoints"]["val"][49][1],
               final_test_task_sd_logged=log["endpoints"]["test"][49][1],
               log_start=log["start"], log_end=log["end"], logged_gpu=log["gpu"],
               wall_seconds=log["wall_seconds"],
               communication_bytes=None, communication_seconds=None,
               communication_missing_reason="not_recorded; prototype_record_counts_are_not_bytes",
               uploaded_prototype_records_total=sum(r["uploaded_prototype_records"] for r in round_rows),
               metric_source=rel(metric), occupancy_source=rel(occupancy), log_source=rel(log_path),
               log_config_line=log["config_line"])
    for r, suffix in ((0, "first"), (49, "final")):
        for scope, rows in (("local", [oi[str(r), "client", str(c)] for c in range(4)]),
                            ("server", [oi[str(r), "server", "server"]])):
            run.update({f"{scope}_{name}_{suffix}": value for name, value in support(rows).items()})
        run["uploaded_prototype_records_" + suffix] = round_rows[r]["uploaded_prototype_records"]
    for scope in ("local", "server"):
        for field in ("cluster_fit_seconds", "collection_seconds"):
            key = scope + "_" + field
            run[key + "_first"] = round_rows[0][key]
            run[key + "_total"] = sum(r[key] for r in round_rows)
            run[key + "_later_round_mean"] = statistics.mean(r[key] for r in round_rows[1:])
    return run, details, round_rows, common_config(cfg), ofields, mfields


def summarize(runs):
    result = []
    baseline = {r["seed"]: r for r in runs if r["k"] == 16}
    fields = ["final_val_auc", "final_test_auc", "wall_seconds", "uploaded_prototype_records_total"]
    for scope in ("local", "server"):
        fields += [scope + "_" + name + "_final" for name in
                   ("nonempty_clusters", "samples", "singleton_fraction", "lt3_fraction", "lt5_fraction")]
        fields += [scope + "_" + name + "_total" for name in ("cluster_fit_seconds", "collection_seconds")]
    for k in KS:
        rows = sorted((r for r in runs if r["k"] == k), key=lambda r: r["seed"])
        require([r["seed"] for r in rows] == list(SEEDS), "Do not select/drop seeds")
        summary = dict(k=k, n_seeds=3, seeds="0;1;2", endpoint="round49_global_auc",
                       sd_definition="sample_sd_ddof1_across_three_joint_model_split_partition_seeds")
        for field in fields:
            summary[field + "_mean"] = statistics.mean(r[field] for r in rows)
            summary[field + "_sd"] = statistics.stdev(r[field] for r in rows)
        diffs = [r["final_test_auc"] - baseline[r["seed"]]["final_test_auc"] for r in rows]
        summary.update(paired_test_auc_minus_k16_mean=statistics.mean(diffs),
                       paired_test_auc_minus_k16_sd=statistics.stdev(diffs),
                       seeds_above_k16=sum(x > 0 for x in diffs), seeds_below_k16=sum(x < 0 for x in diffs))
        result.append(summary)
    return result


def write_csv(path, rows):
    require(rows, f"Refusing empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def review_note(summaries, runs):
    by_k = {r["k"]: r for r in summaries}
    by_case = {(r["k"], r["seed"]): r for r in runs}
    require(by_k[128]["final_test_auc_mean"] > max(by_k[k]["final_test_auc_mean"] for k in (32, 64)),
            "Nonmonotonic AUC example changed; revise note")
    require(by_case[128, 2]["final_test_auc"] > by_case[16, 2]["final_test_auc"],
            "Seed-level counterexample changed; revise note")
    lines = ["# R1.5 本地簇支持审计", "",
             "## 范围与口径",
             "仅纳入 formal_v3/priority_suite/k_occupancy 的 SIDER 15 个完整运行：K=8/16/32/64/128，seed=0/1/2；"
             "alpha=0.5、hetero、MPNN、fingerprint、lambda=0.1、hard matching、4 客户端、50 轮、每轮 200 local steps。"
             "模型/划分/分区 seed 同步变化；除 K、seed 和输出路径外，原日志配置完全一致。未混入 main、tuning、legacy 或 support-aware protocol。",
             "训练客户端均为 4 个，总训练分子数均为 1141；各 seed 的客户端样本数依次为 "
             + "; ".join(f"seed {s}: {runs_by_seed(runs, s)['client_sample_counts_json']}" for s in SEEDS) + "。客户端不是患者人数。",
             "",
             "## 字段可用性",
             "占用率 CSV 没有逐簇 cluster_id/count、分子归属或完整直方图，但直接记录 realized_k、singleton_clusters、"
             "clusters_lt3、clusters_lt5。比例=对应簇数/实际非空簇数；本地先对 4 客户端簇数求和，服务端单列。"
             "未从 mean/median/分位数反推分布。服务端支持数是所聚合本地簇的分子支持总和，不是贡献客户端数或上传原型个数。",
             "15 个运行均核对了 50 轮 AUC、250 条占用记录及日志。实际非空簇为每客户端 K、服务端 K，无空簇；"
             "占用摘要在 50 轮完全相同，首轮未命中缓存、后 49 轮命中。重复轮不是独立重复，也不证明原型向量/归属不变。",
             "",
             "## 三种子结果",
             "final AUC 取 round 49 的 global test，不取 best_test 或最好 seed；SD 为三个 seed 的样本 SD (ddof=1)，"
             "不是日志的任务间 SD，也不是标准误。下列比例是每 seed 比例的均值，CSV 另保留 SD 和逐 seed 计数。",
             "",
             "| K | 本地/服务端非空簇 | 本地 singleton / <3 / <5 | 服务端 singleton / <3 / <5 | final test AUC (mean +/- SD) |",
             "|---:|---:|---:|---:|---:|"]
    for r in summaries:
        fractions = [" / ".join(f"{100*r[f'{scope}_{field}_fraction_final_mean']:.2f}%" for field in ("singleton", "lt3", "lt5")) for scope in ("local", "server")]
        lines.append(f"| {r['k']} | {r['local_nonempty_clusters_final_mean']:.0f}/{r['server_nonempty_clusters_final_mean']:.0f} | {fractions[0]} | {fractions[1]} | {r['final_test_auc_mean']:.4f} +/- {r['final_test_auc_sd']:.4f} |")
    lines += ["", "## 能答与不能答",
              "能答：该协议下大 K 的本地低支持簇明显增多，服务端仍保留不少低支持簇，支持审稿人对小样本原型可靠性的担忧。"
              "但服务端低支持比例并非随 K 单调变化，AUC 与跨 seed SD 也不单调。K=128 的平均 AUC 高于 K=32/64；"
              f"seed 2 的 K=128 AUC 为 {by_case[128, 2]['final_test_auc']:.4f}，高于 K=16 的 {by_case[16, 2]['final_test_auc']:.4f}。"
              "负结果和反例均保留，不能写成“大 K 必然使性能下降”。",
              "不能答：缺少逐簇计数明细、原型向量、bootstrap/重采样方差及成员映射，无法直接量化原型估计不稳定、"
              "分子受低支持簇影响的完整分布、跨客户端贡献或机制因果。仅三个联合变化 seed，不能把 AUC SD 解释为纯初始化方差，"
              "本审计不作显著性结论或外推其他数据集/协议。这里本地与全局 K 同时变化，也不能分离两者效应；未据测试集重新选择超参数。",
              "", "## 时间与通信",
              "run_audit.csv 保留日志 START/END 差值（整次进程墙钟秒数）及首轮、全程、后续轮均值的聚类/收集计时。"
              "本地计时按客户端相加，不是并行关键路径；collection_seconds 包含聚类时间，不能再与 cluster_fit_seconds 相加。"
              "本地计时在构建原型列表前结束，服务端为聚合计时；不是纯训练或网络时间。不同 GPU/并发负载未受控，不作效率优劣推断。",
              "三种子平均进程墙钟秒数按 K=8/16/32/64/128 依次为 "
              + "/".join(f"{by_k[k]['wall_seconds_mean']:.0f}" for k in KS)
              + "，也不单调增加；聚类计时增加不能被改写成实测总运行时间必然增加。",
              "日志通信可核对 50 轮、每轮 4 客户端、每轮上传 4K 条本地原型记录（全程 200K 条），"
              "以及每轮聚合 K 个全局原型；后者不是实测下行次数。没有通信字节量或网络耗时，CSV 留空并注明原因，不根据维度估算。",
              "", "## 文件与复现",
              "run_audit.csv：15 个 K/seed 的支持、AUC、计时、通信代理计数和来源；k_summary.csv：三种子均值/样本 SD，"
              "并列出相同 seed 对 K=16 的描述性差值。support_by_scope.csv：首/末轮逐客户端与服务端的直接记录（150 行）；"
              "round_audit.csv：750 个运行轮次的日志核对与计时。schema_and_protocol.json、source_inventory.csv 保存字段能力、协议和输入 SHA-256。",
              "支持计数/计时与日志任务间 SD 的语义参考 audit_20260910/source_snapshot/{client,fedavg_api}.py 的只读代码；"
              "它是后续快照，不冒充 9 月 1 日运行的精确代码版本。实验数值仅来自指定 formal_v3 套件。",
              "", "```bash",
              f"cd {ROOT}",
              f"{sys.executable} -B scripts/reviewer_revision/audit_cluster_support_review.py --self-test",
              f"{sys.executable} -B scripts/reviewer_revision/audit_cluster_support_review.py",
              "```",
              "", "仅标准库本地解析；不导入训练模块、不连服务器、不改正文或已有源文件。缺失/重复运行、协议不一致、计数异常会中止，不静默择优。", ""]
    return "\n".join(lines)


def runs_by_seed(runs, seed):
    return next(r for r in runs if r["k"] == 8 and r["seed"] == seed)


def audit():
    inventory = {}
    _, manifest = read_csv(SOURCE / "revision_priority_manifest_v3.csv", inventory, "selection_manifest")
    _, statuses = read_csv(SOURCE / "summaries/expected_run_status.csv", inventory, "existing_summary_crosscheck")
    selected = [r for r in manifest if r["suite"] == "k_occupancy"]
    expected_ids = {f"k_occupancy_SIDER_a0p5_hetero_seed{s}_fedavg_proto_mpnn_fingerprint_k{k}" for k in KS for s in SEEDS}
    mi = indexed(selected, ("case_id",))
    require(set(mi) == {(x,) for x in expected_ids}, "Expected exactly the 15 requested cases")
    require({p.name for p in (SOURCE / "priority_suite/k_occupancy").iterdir()} == expected_ids, "Unexpected run directory")
    si = indexed([r for r in statuses if r["suite"] == "k_occupancy"], ("case_id",))
    require(set(si) == set(mi), "Manifest/status coverage mismatch")
    runs, details, rounds, common, schemas = [], [], [], None, None
    for m in sorted(selected, key=lambda r: (int(r["k"]), int(r["seed"]))):
        run, detail, round_rows, cfg, ofields, mfields = audit_case(m, si[(m["case_id"],)], inventory)
        if common is None:
            common, schemas = cfg, (ofields, mfields)
        require(cfg == common, "Non-K/non-seed logged configuration differs across cases")
        require(schemas == (ofields, mfields), "CSV schemas differ across cases")
        runs.append(run)
        details.extend(detail)
        rounds.extend(round_rows)
    for seed in SEEDS:
        require(len({r["client_sample_counts_json"] for r in runs if r["seed"] == seed}) == 1, "Partition sizes change with K")
    # This frozen-note audit must not carry current conclusions onto changed data.
    require(all(r["occupancy_stats_unchanged_all_rounds"] for r in runs), "Occupancy changed across rounds; revise summary interpretation")
    require(all(r["local_nonempty_clusters_final"] == 4*r["k"] and r["server_nonempty_clusters_final"] == r["k"] for r in runs), "Empty clusters present; revise note")
    require(all(r["local_samples_final"] == 1141 for r in runs), "Training sample count changed")
    require(all(r["local_cache_hits"] == (0 if r["round"] == 0 else 4)
                and r["server_cache_hit"] == (r["round"] > 0) for r in rounds), "Unexpected cache behavior")
    summaries = summarize(runs)
    for run in runs:
        summary = next(r for r in summaries if r["k"] == run["k"])
        run["k_final_test_auc_mean"] = summary["final_test_auc_mean"]
        run["k_final_test_auc_sample_sd"] = summary["final_test_auc_sd"]
    remember(Path(__file__), inventory, "audit_script")
    for name in ("client.py", "fedavg_api.py"):
        remember(BASE / "audit_20260910/source_snapshot" / name, inventory, "later_read_only_semantics_reference_not_exact_run_version")
    provenance = dict(
        audit_date="2026-09-16", runs=len(runs), metric_rows=len(rounds), occupancy_rows=3750,
        endpoint="final round 49 global val/test AUC; never best_test", sd="statistics.stdev, ddof=1, n=3 seeds",
        local_fraction="sum(client threshold cluster counts) / sum(client realized_k), per seed; then seed mean/SD",
        server_fraction="server threshold cluster counts / server realized_k, per seed; then seed mean/SD",
        per_cluster_counts_available=False, exact_threshold_counts_available=True,
        missing=["cluster_id/count vectors", "molecule assignments", "prototype embeddings/variance",
                 "number of contributing clients per server cluster", "communication bytes", "network time"],
        occupancy_fields=schemas[0], metric_fields=schemas[1], common_logged_configuration=common,
        allowed_configuration_variation=sorted(VARIABLE_CONFIG),
        source_paths_relative_to=str(ROOT), source_version_caveat="20260910 source snapshot is semantic context, not exact 20260901 run version",
        time_caveat="Process wall time; summed client collection is not parallel elapsed time; fit is included in collection; not network timers",
        communication_caveat="Uploaded prototype records only; not measured bytes or downlink transmissions",
        log_std_caveat="Final logged task SD (np.std, ddof=0 in later source snapshot) is separate from across-seed sample SD",
        units=dict(auc="0_to_1", fractions="0_to_1_not_percent", timers="seconds",
                   samples="training_molecules_not_patients", prototype_records="counts_not_bytes"),
        training_started=False, server_connected=False, manuscript_modified=False,
        python_executable=sys.executable, python_version=sys.version.split()[0])
    note = review_note(summaries, runs)
    for item in inventory.values():
        require(digest(ROOT / item["path"]) == item["sha256"], "Input changed during audit")
    require(OUTPUT.resolve() == OUTPUT, "Output directory must not be a symlink")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {"run_audit.csv": runs, "support_by_scope.csv": details, "round_audit.csv": rounds,
               "k_summary.csv": summaries, "source_inventory.csv": sorted(inventory.values(), key=lambda r: r["path"])}
    for name in list(outputs) + ["schema_and_protocol.json", "REVIEW_NOTE.md"]:
        require(not (OUTPUT / name).is_symlink(), "Refusing output symlink")
    for name, rows in outputs.items():
        write_csv(OUTPUT / name, rows)
    (OUTPUT / "schema_and_protocol.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "REVIEW_NOTE.md").write_text(note, encoding="utf-8")
    print(f"Verified {len(runs)} runs, {len(rounds)} rounds, 3750 occupancy records. No training/network.")
    print(f"Output: {OUTPUT}")


class AuditTests(unittest.TestCase):
    def row(self, n=4, singleton=1, lt3=2, lt5=3):
        return dict(realized_k=n, samples=20 if n else 0, requested_k=8, min_count=1,
                    singleton_clusters=singleton, clusters_lt3=lt3, clusters_lt5=lt5)

    def test_pooled_denominator(self):
        result = support([self.row(), self.row(n=2, singleton=0, lt3=1, lt5=2)])
        self.assertEqual(result["singleton_fraction"], 1/6)
        self.assertEqual(result["lt3_fraction"], 3/6)

    def test_missing_numerator_not_inferred(self):
        row = self.row()
        del row["clusters_lt3"]
        row["mean_count"] = 5
        with self.assertRaises(KeyError):
            support([row])

    def test_zero_denominator_is_missing(self):
        self.assertIsNone(support([self.row(n=0, singleton=0, lt3=0, lt5=0)])["lt3_fraction"])

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            support([self.row(singleton=4, lt3=2)])
        with self.assertRaises(ValueError):
            integer("1.5")
        with self.assertRaises(ValueError):
            number("nan")

    def test_final_not_best_or_file_order(self):
        rows = [dict(round=str(r), scope="global", test=r/100, best_test=0.99) for r in range(50)]
        self.assertEqual(final_metric(list(reversed(rows)))["test"], 0.49)

    def test_missing_and_duplicate_rounds(self):
        rows = [dict(round=str(r), scope="global") for r in range(50)]
        for invalid in (rows[:-1], rows + [rows[-1]]):
            with self.assertRaises(ValueError):
                final_metric(invalid)

    def test_protocol_rejected(self):
        config = dict(FIXED_CONFIG, seed=0, model_seed=0, split_seed=0, partition_seed=0,
                      topoproto_clusters=8, topoproto_global_clusters=8)
        check_config(config, 8, 0)
        config["lambda_proto"] = 0.5
        with self.assertRaises(ValueError):
            check_config(config, 8, 0)

    def test_sample_sd(self):
        self.assertEqual(statistics.stdev([1, 2, 3]), 1)
        self.assertNotEqual(statistics.stdev([1, 2, 3]), statistics.pstdev([1, 2, 3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run in-memory tests without writing files")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        audit()
