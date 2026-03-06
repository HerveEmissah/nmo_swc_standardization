"""
Filename: app.py
Author: Herve Emissah
Created: 2023-10-14
Description: MPI-parallel tree-level neural morphology classification pipeline ran on HPC.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mpi4py import MPI
from datetime import datetime
import logging
import glob
import sys
import csv
import math
from typing import Set, Dict, Any, Optional, List, Tuple

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Recursion limit to handle deep graph structures
sys.setrecursionlimit(3000)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Feature columns for tree feature vector
# ============================================================
FEATURE_COLUMNS = [
    "node_count",
    "tip_count",
    "bifurcations",
    "max_euclid",
    "mean_euclid",
    "max_path_length",
    "total_length",
    "sholl_sum",
    "elongation",
    "principal_axis_x",
    "principal_axis_y",
    "principal_axis_z",
]

# ============================================================
# Geometry utilities
# ============================================================
def label_to_int(label):
    return 0 if label == "apical" else 1 if label == "basal" else 2

def label_to_string(label):
    return ["apical", "basal", "other"][label]

def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def _safe_float(v, default=0.0):
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return float(default)
    except Exception:
        return float(default)

def _node_xyz(G: nx.DiGraph, n: int) -> np.ndarray:
    return np.array(
        [
            _safe_float(G.nodes[n].get("x", 0.0), 0.0),
            _safe_float(G.nodes[n].get("y", 0.0), 0.0),
            _safe_float(G.nodes[n].get("z", 0.0), 0.0),
        ],
        dtype=float,
    )

def _edge_len(G: nx.DiGraph, u: int, v: int) -> float:
    return float(np.linalg.norm(_node_xyz(G, v) - _node_xyz(G, u)))

# ============================================================
# Model
# ============================================================
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)
        return out

# ============================================================
# Robust SWC reader
# ============================================================
def read_swc_file(filename: str) -> nx.DiGraph:
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["node_id", "node_type", "x", "y", "z", "radius", "parent"],
        engine="python",
        dtype=str,
        encoding="ISO-8859-1",
    )

    df = df.dropna(how="all").copy()

    cols = ["node_id", "node_type", "x", "y", "z", "radius", "parent"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    required = ["node_id", "node_type", "parent", "x", "y", "z"]
    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError(f"{os.path.basename(filename)}: SWC empty after removing invalid rows.")

    df["node_id"] = df["node_id"].round().astype(int)
    df["node_type"] = df["node_type"].round().astype(int)
    df["parent"] = df["parent"].round().astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)

    df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["radius"] = df["radius"].fillna(1.0).astype(float)

    node_ids = set(df["node_id"].tolist())
    bad_parent_mask = (df["parent"] != -1) & (~df["parent"].isin(node_ids))
    if bad_parent_mask.any():
        df = df.loc[~bad_parent_mask].copy()

    if df.empty:
        raise ValueError(f"{os.path.basename(filename)}: SWC empty after removing rows with missing parents.")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        parent = int(row["parent"])
        G.add_node(
            nid,
            node_type=int(row["node_type"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            radius=float(row["radius"]),
            parent=parent,
        )

    for nid, data in G.nodes(data=True):
        parent = int(data.get("parent", -1))
        if parent != -1 and parent in G:
            G.add_edge(parent, nid)

    return G

# ============================================================
# Apical/basal rule helpers
# ============================================================
def ensure_soma_node(graph: nx.DiGraph):
    root_nodes = [n for n in graph.nodes if int(graph.nodes[n].get("parent", 999999)) == -1]
    if not root_nodes:
        return None

    soma_node = sorted(root_nodes)[0]
    if int(graph.nodes[soma_node].get("node_type", -1)) != 1:
        graph.nodes[soma_node]["node_type"] = 1
    return soma_node

def build_soma_children_trees(graph: nx.DiGraph, soma_node: int) -> Tuple[List[List[int]], List[int]]:
    trees = []
    tree_root_map = []
    for child in graph.successors(soma_node):
        stack = [child]
        nodeset: Set[int] = set()
        while stack:
            cur = stack.pop()
            if cur == soma_node or cur in nodeset:
                continue
            nodeset.add(cur)
            stack.extend(list(graph.successors(cur)))
        if len(nodeset) > 0:
            trees.append(sorted(nodeset))
            tree_root_map.append(child)
    return trees, tree_root_map

def compute_tree_topology_paths(G: nx.DiGraph, root: int, tree_nodes: Set[int]) -> Tuple[float, float, int]:
    total_length = 0.0
    for u, v in G.edges():
        if u in tree_nodes and v in tree_nodes:
            total_length += _edge_len(G, u, v)

    tip_count = 0
    for n in tree_nodes:
        child_count = sum(1 for c in G.successors(n) if c in tree_nodes)
        if child_count == 0:
            tip_count += 1

    max_path = 0.0
    stack = [(root, 0.0)]
    while stack:
        node, acc = stack.pop()
        children = [c for c in G.successors(node) if c in tree_nodes]
        if not children:
            if acc > max_path:
                max_path = acc
        else:
            for c in children:
                stack.append((c, acc + _edge_len(G, node, c)))
    return float(max_path), float(total_length), int(tip_count)

def calculate_sholl_value(G: nx.DiGraph, tree_nodes: Set[int], soma_node: int, radial_step=10.0) -> float:
    soma_coord = _node_xyz(G, soma_node)
    distances = [float(np.linalg.norm(_node_xyz(G, n) - soma_coord)) for n in tree_nodes]
    if not distances:
        return 0.0

    maxd = max(distances)
    if maxd <= 0.0:
        return 0.0

    bins = np.arange(0.0, maxd + radial_step, radial_step)
    counts = np.histogram(distances, bins=bins)[0]
    ring_indices = np.arange(1, len(counts) + 1)
    weighted = float(np.sum(counts * ring_indices))
    return weighted

def compute_tree_features_aligned(
    G: nx.DiGraph,
    soma: int,
    root: int,
    tree_nodes: Set[int],
    radial_step: float = 10.0,
) -> Dict[str, Any]:
    coords = np.array([_node_xyz(G, n) for n in tree_nodes], dtype=float) if tree_nodes else np.zeros((0, 3))
    soma_coord = _node_xyz(G, soma)

    dists = [float(np.linalg.norm(_node_xyz(G, n) - soma_coord)) for n in tree_nodes]
    max_euclid = float(max(dists)) if dists else 0.0
    mean_euclid = float(np.mean(dists)) if dists else 0.0

    max_path_length, total_length, tip_count = compute_tree_topology_paths(G, root, tree_nodes)
    sholl_sum = calculate_sholl_value(G, tree_nodes, soma, radial_step=radial_step)
    bifurcations = sum(1 for n in tree_nodes if sum(1 for c in G.successors(n) if c in tree_nodes) >= 2)
    node_count = int(len(tree_nodes))

    elongation = 0.0
    principal_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    try:
        if coords.shape[0] >= 2:
            centered = coords - coords.mean(axis=0)
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            s0 = float(S[0]) if len(S) > 0 else 0.0
            s1 = float(S[1]) if len(S) > 1 else 1e-9
            elongation = float(s0 / (s1 + 1e-12))
            principal_axis = Vt[0] if Vt.shape[0] >= 1 else principal_axis
    except Exception:
        pass

    return {
        "node_count": node_count,
        "tip_count": int(tip_count),
        "bifurcations": int(bifurcations),
        "max_euclid": float(max_euclid),
        "mean_euclid": float(mean_euclid),
        "max_path_length": float(max_path_length),
        "total_length": float(total_length),
        "sholl_sum": float(sholl_sum),
        "elongation": float(elongation),
        "principal_axis_x": float(principal_axis[0]),
        "principal_axis_y": float(principal_axis[1]),
        "principal_axis_z": float(principal_axis[2]),
    }

def apical_candidate_filter(
    feats: Dict[int, Dict[str, Any]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    debug: bool = False,
    stub_node_max: int = 50,
) -> List[int]:
    if not eligible_idx:
        return []

    max_nodes = max(feats[i]["node_count"] for i in eligible_idx)
    major_min_nodes = max(50, int(round(0.25 * max_nodes)))
    stubs, majors, minors = [], [], []

    for i in eligible_idx:
        m = feats[i]
        is_unbranched = (m.get("bifurcations", 0) == 0 and m.get("tip_count", 0) == 1)
        is_stub = is_unbranched and (m["node_count"] < stub_node_max)
        if is_stub:
            stubs.append(i)
            continue
        if m["node_count"] >= major_min_nodes:
            majors.append(i)
        else:
            minors.append(i)

    candidates = majors if majors else [i for i in eligible_idx if i not in stubs]
    if not candidates:
        candidates = list(eligible_idx)

    if debug:
        logger.info(f"[apical-gate] max_nodes={max_nodes} major_min_nodes={major_min_nodes}")
        if stubs:
            logger.info(f"[apical-gate] excluded_stubs: {[(i, tree_root_map[i], feats[i]['node_count']) for i in stubs]}")
        if minors:
            logger.info(f"[apical-gate] excluded_minors: {[(i, tree_root_map[i], feats[i]['node_count']) for i in minors]}")
        logger.info(f"[apical-gate] candidates: {[(i, tree_root_map[i], feats[i]['node_count']) for i in candidates]}")

    return candidates

def select_apical_tree_feature_fallback(
    graph: nx.DiGraph,
    trees: List[List[int]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    soma_node: int,
    radial_step: float = 10.0,
    debug: bool = False,
) -> int:
    feats: Dict[int, Dict[str, Any]] = {}
    for i in eligible_idx:
        feats[i] = compute_tree_features_aligned(
            graph,
            soma=soma_node,
            root=tree_root_map[i],
            tree_nodes=set(trees[i]),
            radial_step=radial_step,
        )

    candidates = apical_candidate_filter(feats, eligible_idx, tree_root_map, debug=debug)

    def score(i: int) -> float:
        f = feats[i]
        s = 0.0
        s += math.log1p(f.get("max_path_length", 0.0)) * 1.3
        s += math.log1p(f.get("total_length", 0.0)) * 1.0
        s += math.log1p(f.get("sholl_sum", 0.0)) * 0.8
        s += math.log1p(f.get("node_count", 0.0)) * 0.6
        s += math.log1p(f.get("max_euclid", 0.0)) * 0.7
        axis_max = max(
            abs(float(f.get("principal_axis_x", 0.0))),
            abs(float(f.get("principal_axis_y", 0.0))),
            abs(float(f.get("principal_axis_z", 0.0))),
        )
        s += float(f.get("elongation", 0.0)) * 0.6
        s += float(axis_max) * 0.5
        return float(s)

    apical = max(candidates, key=score)
    return apical

def _local_z(values: List[float]) -> List[float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return []
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd <= 1e-12:
        return [0.0] * len(values)
    return list((a - mu) / sd)

def select_apical_by_morphology(
    graph: nx.DiGraph,
    trees: List[List[int]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    soma_node: int,
    radial_step: float = 10.0,
    debug: bool = False,
) -> int:
    feats: Dict[int, Dict[str, Any]] = {}
    for i in eligible_idx:
        feats[i] = compute_tree_features_aligned(
            graph,
            soma=soma_node,
            root=tree_root_map[i],
            tree_nodes=set(trees[i]),
            radial_step=radial_step,
        )

    candidates = apical_candidate_filter(feats, eligible_idx, tree_root_map, debug=debug)
    if not candidates:
        candidates = list(eligible_idx)

    max_path_vals, max_euc_vals = [], []
    sholl_vals, tot_len_vals, node_cnt_vals = [], [], []
    elong_vals, axis_max_vals = [], []

    for i in candidates:
        f = feats[i]
        max_path_vals.append(float(f.get("max_path_length", 0.0)))
        max_euc_vals.append(float(f.get("max_euclid", 0.0)))
        sholl_vals.append(float(f.get("sholl_sum", 0.0)))
        tot_len_vals.append(float(f.get("total_length", 0.0)))
        node_cnt_vals.append(float(f.get("node_count", 0.0)))
        elong_vals.append(float(f.get("elongation", 0.0)))
        axis_max_vals.append(max(
            abs(float(f.get("principal_axis_x", 0.0))),
            abs(float(f.get("principal_axis_y", 0.0))),
            abs(float(f.get("principal_axis_z", 0.0))),
        ))

    z_max_path = _local_z([math.log1p(x) for x in max_path_vals])
    z_max_euc  = _local_z([math.log1p(x) for x in max_euc_vals])
    z_sholl    = _local_z([math.log1p(x) for x in sholl_vals])
    z_tot_len  = _local_z([math.log1p(x) for x in tot_len_vals])
    z_node_cnt = _local_z([math.log1p(x) for x in node_cnt_vals])
    z_elong    = _local_z(elong_vals)
    z_axis_max = _local_z(axis_max_vals)

    W = {
        "max_path": 1.5,
        "max_euc":  1.2,
        "sholl":    0.8,
        "tot_len":  0.7,
        "node_cnt": 0.5,
        "elong":    0.8,
        "axis_max": 0.6,
    }

    idx_to_score: Dict[int, float] = {}
    for k, i in enumerate(candidates):
        score = (
            W["max_path"] * z_max_path[k]
            + W["max_euc"]  * z_max_euc[k]
            + W["sholl"]    * z_sholl[k]
            + W["tot_len"]  * z_tot_len[k]
            + W["node_cnt"] * z_node_cnt[k]
            + W["elong"]    * z_elong[k]
            + W["axis_max"] * z_axis_max[k]
        )
        idx_to_score[i] = float(score)

    best = max(candidates, key=lambda i: idx_to_score[i])
    return best

def assign_apical_basal_rule(
    graph: nx.DiGraph,
    radial_step: float = 10.0,
    debug: bool = False,
) -> Dict[str, Any]:
    soma_node = ensure_soma_node(graph)
    if soma_node is None:
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    trees, tree_root_map = build_soma_children_trees(graph, soma_node)
    if not trees:
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    eligible_idx = []
    for i, tree_nodes in enumerate(trees):
        node_types = [_safe_int(graph.nodes[n].get("node_type", -1)) for n in tree_nodes]
        has_dendrite = any(nt in (3, 4) for nt in node_types)
        if has_dendrite:
            eligible_idx.append(i)

    if not eligible_idx:
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    apical_idx = select_apical_by_morphology(
        graph=graph,
        trees=trees,
        eligible_idx=eligible_idx,
        tree_root_map=tree_root_map,
        soma_node=soma_node,
        radial_step=radial_step,
        debug=debug,
    )

    if apical_idx is None:
        apical_idx = select_apical_tree_feature_fallback(
            graph=graph,
            trees=trees,
            eligible_idx=eligible_idx,
            tree_root_map=tree_root_map,
            soma_node=soma_node,
            radial_step=radial_step,
            debug=debug,
        )

    changed = 0
    apical_set = {apical_idx}
    basal_set = set(eligible_idx) - apical_set

    for i in basal_set:
        for n in trees[i]:
            if n == soma_node:
                continue
            cur = _safe_int(graph.nodes[n].get("node_type", -1))
            if cur != 3:
                graph.nodes[n]["node_type"] = 3
                changed += 1

    for n in trees[apical_idx]:
        if n == soma_node:
            continue
        cur = _safe_int(graph.nodes[n].get("node_type", -1))
        if cur != 4:
            graph.nodes[n]["node_type"] = 4
            changed += 1

    bad_basal = 0
    for i in sorted(basal_set):
        bad_nodes = [n for n in trees[i] if _safe_int(graph.nodes[n].get("node_type", -1)) == 4]
        if bad_nodes:
            bad_basal += 1

    return {
        "apical_index": apical_idx,
        "apical_root": tree_root_map[apical_idx],
        "changed_nodes": changed,
        "bad_basal_trees": bad_basal,
    }

# ============================================================
# Training / evaluation
# ============================================================
def train(model, dataloader, optimizer, criterion, train_metadata):
    model.train()
    total_loss = 0.0
    correct_files = 0
    total_files = 0

    for features, labels, file_name in dataloader:
        file_name = file_name[0] if isinstance(file_name, (list, tuple)) else file_name
        file_name = file_name.strip().lower()

        features = features.view(-1, features.size(-1))
        labels = labels.view(-1)

        try:
            _ = next(
                meta for meta in train_metadata
                if os.path.basename(meta["file_name"]).strip().lower() == file_name
            )
        except StopIteration:
            logger.error(
                f"File {file_name} not found in train_metadata. "
                f"Available files: {[os.path.basename(meta['file_name']) for meta in train_metadata]}"
            )
            continue

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        predicted[labels == 2] = 2

        non_other_idx = (labels != 2).nonzero(as_tuple=True)[0]
        if non_other_idx.numel() > 0:
            margins = outputs[non_other_idx, 0] - outputs[non_other_idx, 1]
            max_idx = non_other_idx[margins.argmax()]
            predicted[non_other_idx] = 1
            predicted[max_idx] = 0

        if torch.equal(predicted, labels):
            correct_files += 1
        total_files += 1

    train_accuracy = 100 * correct_files / total_files if total_files > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss, train_accuracy

def evaluate(model, dataloader, criterion, eval_metadata):
    model.eval()
    total_loss = 0.0
    correct_files = 0
    total_files = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels, file_name in dataloader:
            features = features.view(-1, features.size(-1))
            labels = labels.view(-1)

            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predicted[labels == 2] = 2

            non_other_idx = (labels != 2).nonzero(as_tuple=True)[0]
            if non_other_idx.numel() > 0:
                margins = outputs[non_other_idx, 0] - outputs[non_other_idx, 1]
                max_idx = non_other_idx[margins.argmax()]
                predicted[non_other_idx] = 1
                predicted[max_idx] = 0

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if torch.equal(predicted, labels):
                correct_files += 1
            total_files += 1

    eval_accuracy = 100 * correct_files / total_files if total_files > 0 else 0
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {eval_accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    model.train()
    return avg_loss, eval_accuracy, precision, recall, f1_score

def test_evaluate(model, test_metadata, test_loader, cycle):
    model.eval()
    total_loss = 0.0
    correct_files = 0
    total_files = 0
    criterion = torch.nn.CrossEntropyLoss()
    incorrect_predictions = set()
    last_file_name = None
    all_labels = []
    all_predictions = []

    with open(f"cycle{cycle+1}_neuron_test_accuracy.csv", "w", newline='') as csvfile:
        csvfile.write(f"{'file_name'.ljust(50)}{'actual_tree_type'.ljust(20)}{'predicted_tree_type'.ljust(20)}{'tree_nodes'}\n")

        with torch.no_grad():
            for features, labels, file_name_batch in test_loader:
                features = features.view(-1, features.size(-1))
                labels = labels.view(-1)

                file_name_str = file_name_batch[0] if isinstance(file_name_batch, (list, tuple)) else file_name_batch
                file_metadata = next(meta for meta in test_metadata if meta["file_name"] == file_name_str)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                predictions[labels == 2] = 2

                non_other_idx = (labels != 2).nonzero(as_tuple=True)[0]
                if non_other_idx.numel() > 0:
                    margins = outputs[non_other_idx, 0] - outputs[non_other_idx, 1]
                    max_idx = non_other_idx[margins.argmax()]
                    predictions[non_other_idx] = 1
                    predictions[max_idx] = 0

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                for i in range(predictions.size(0)):
                    predicted_label = label_to_string(predictions[i].item())
                    actual_label = label_to_string(labels[i].item())

                    tree_nodes = file_metadata["trees"][i]["tree_nodes"]
                    tree_nodes_str = " -> ".join(map(str, tree_nodes))

                    if last_file_name and file_name_str != last_file_name:
                        csvfile.write("\n")

                    csvfile.write(f"{file_name_str.ljust(50)}{actual_label.ljust(20)}{predicted_label.ljust(20)}{tree_nodes_str}\n")

                    if predicted_label != actual_label:
                        incorrect_predictions.add(file_name_str)

                    last_file_name = file_name_str

                if torch.equal(predictions, labels):
                    correct_files += 1
                total_files += 1

        accuracy = 100 * correct_files / total_files if total_files > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        csvfile.write("\n\n")
        csvfile.write("Summary Statistics:\n")
        csvfile.write(f"Total Number of Neurons: {total_files}\n")
        csvfile.write(f"Total Number of Neurons Correctly Predicted: {correct_files}\n")
        csvfile.write(f"Total Number of Neurons Incorrectly Predicted: {total_files - correct_files}\n")
        csvfile.write("\n")
        csvfile.write(f"Overall Test Accuracy: {accuracy:.2f}%\n")

    logger.info(f"Final Test Accuracy: {accuracy:.2f}% | Average Loss: {avg_loss:.4f}")
    logger.info(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    model.train()
    return avg_loss, accuracy, precision, recall, f1_score

# ============================================================
# DataLoader for file-level batches
# ============================================================
def create_file_level_dataloader(flattened_metadata, batch_size=1):
    """
    Create DataLoader where each batch corresponds to all trees of a file.
    Uses full feature vector per tree.
    """
    file_features = []
    file_labels = []
    file_names = []

    for file_data in flattened_metadata:
        file_name = file_data["file_name"]
        trees = file_data["trees"]

        if not trees:
            continue

        tree_features = []
        tree_labels = []

        for tree in trees:
            feats = tree.get("features", {})
            vec = [_safe_float(feats.get(col, 0.0), 0.0) for col in FEATURE_COLUMNS]
            tree_features.append(torch.tensor(vec, dtype=torch.float32))
            tree_labels.append(label_to_int(tree["tree_type"]))

        file_features.append(torch.stack(tree_features))
        file_labels.append(torch.tensor(tree_labels, dtype=torch.long))
        file_names.append(file_name)

    dataset = list(zip(file_features, file_labels, file_names))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============================================================
# Tree classification using assign_apical_basal_rule
# ============================================================
def classify_trees(graph, source_file_path):
    file_name = os.path.basename(source_file_path)

    soma_node = ensure_soma_node(graph)
    if soma_node is None:
        return None

    trees, tree_root_map = build_soma_children_trees(graph, soma_node)
    if not trees:
        return None

    graph_for_rule = graph.copy()
    rule_result = assign_apical_basal_rule(graph_for_rule, radial_step=10.0, debug=False)
    apical_idx = rule_result.get("apical_index", None)

    file_classification = {"file_name": file_name, "trees": []}

    logger.info(f"*********File: {file_name}")

    for i, tree_nodes in enumerate(trees):
        node_types = [_safe_int(graph.nodes[n].get("node_type", -1)) for n in tree_nodes]
        has_dendrite = any(nt in (3, 4) for nt in node_types)

        if not has_dendrite:
            tree_type = "other"
            feats = {col: 0.0 for col in FEATURE_COLUMNS}
            sholl_value = 0.0
        else:
            tree_type = "apical" if i == apical_idx else "basal"
            feats = compute_tree_features_aligned(
                graph,
                soma=soma_node,
                root=tree_root_map[i],
                tree_nodes=set(tree_nodes),
                radial_step=10.0,
            )
            sholl_value = feats["sholl_sum"]

            logger.info(
                f"Tree idx={i}, root={tree_root_map[i]}, "
                f"tree_type={tree_type}, sholl_sum={sholl_value:.4f}, "
                f"node_count={feats['node_count']}, max_path={feats['max_path_length']:.4f}"
            )

        file_classification["trees"].append({
            "tree_nodes": sorted(tree_nodes),
            "tree_type": tree_type,
            "sholl_value": sholl_value,
            "features": feats,
        })

        logger.info("*******************")

    return file_classification if file_classification["trees"] else None

# ============================================================
# Main execution
# ============================================================
main_start_time = datetime.now()

if rank == 0:
    directory = 'pyramidals_1_Apical'
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.swc')]
    logger.info(f"Found {len(all_files)} SWC files in the directory.")

    np.random.shuffle(all_files)

    num_total = len(all_files)
    num_train = int(num_total * 0.80)
    num_val = int(num_total * 0.10)
    num_test = num_total - (num_train + num_val)

    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]

    logger.info(f"Rank 0: Data Split -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    chunk_size_train = len(train_files) // size
    remainder_train = len(train_files) % size

    chunk_size_val = len(val_files) // size
    remainder_val = len(val_files) % size

    chunk_size_test = len(test_files) // size
    remainder_test = len(test_files) % size

    train_files_chunks = [
        train_files[i * chunk_size_train + min(i, remainder_train):(i + 1) * chunk_size_train + min(i + 1, remainder_train)]
        for i in range(size)
    ]
    val_files_chunks = [
        val_files[i * chunk_size_val + min(i, remainder_val):(i + 1) * chunk_size_val + min(i + 1, remainder_val)]
        for i in range(size)
    ]
    test_files_chunks = [
        test_files[i * chunk_size_test + min(i, remainder_test):(i + 1) * chunk_size_test + min(i + 1, remainder_test)]
        for i in range(size)
    ]
else:
    train_files_chunks = val_files_chunks = test_files_chunks = None

train_files_chunk = comm.scatter(train_files_chunks, root=0)
val_files_chunk = comm.scatter(val_files_chunks, root=0)
test_files_chunk = comm.scatter(test_files_chunks, root=0)

logger.info(
    f"Rank {rank}: Received chunk of files for training ({len(train_files_chunk)} files), "
    f"validation ({len(val_files_chunk)} files), and testing ({len(test_files_chunk)} files)."
)

def process_files(file_list):
    metadata_list = []
    for file in file_list:
        try:
            graph = read_swc_file(file)
            file_classification = classify_trees(graph, file)
            if file_classification:
                metadata_list.append(file_classification)
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}", exc_info=True)
    return metadata_list

train_metadata = process_files(train_files_chunk)
val_metadata = process_files(val_files_chunk)
test_metadata = process_files(test_files_chunk)

train_metadata = comm.gather(train_metadata, root=0)
val_metadata = comm.gather(val_metadata, root=0)
test_metadata = comm.gather(test_metadata, root=0)

if rank == 0:
    train_metadata = [item for sublist in train_metadata for item in sublist]
    val_metadata = [item for sublist in val_metadata for item in sublist]
    test_metadata = [item for sublist in test_metadata for item in sublist]

    logger.info(
        f"Rank 0: Combined metadata for training ({len(train_metadata)}), "
        f"validation ({len(val_metadata)}), and testing ({len(test_metadata)})."
    )
else:
    train_metadata = val_metadata = test_metadata = None

train_metadata = comm.bcast(train_metadata, root=0)
val_metadata = comm.bcast(val_metadata, root=0)
test_metadata = comm.bcast(test_metadata, root=0)

logger.info(f"Rank {rank}: Using Train: {len(train_metadata)}, Val: {len(val_metadata)}, Test: {len(test_metadata)}")

num_cycles = 10
num_epochs = 50
test_accuracies = []

if rank == 0:
    model_files = glob.glob('best_model_cycle*.pth')
    for model_file in model_files:
        os.remove(model_file)
        logger.info(f"Deleted old model: {model_file}")

comm.Barrier()

# ============================================================
# Initialize Model and Optimizer
# ============================================================
apical_weight = 3.0
basal_weight = 1.0
other_weight = 1.0
class_weights = torch.tensor([apical_weight, basal_weight, other_weight], dtype=torch.float32)

model = GCN(input_dim=len(FEATURE_COLUMNS), hidden_dim=1024, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

train_loader = create_file_level_dataloader(train_metadata)
val_loader = create_file_level_dataloader(val_metadata)
test_loader = create_file_level_dataloader(test_metadata)

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

cycles_per_rank = list(range(num_cycles))[rank::size]
logger.info(f"Rank {rank} will process cycles: {cycles_per_rank}")

last_test_accuracy = 0.0

for cycle in cycles_per_rank:
    logger.info(f"\nRank {rank}: Starting Cycle {cycle + 1}...")

    np.random.shuffle(train_metadata)
    np.random.shuffle(val_metadata)
    np.random.shuffle(test_metadata)
    logger.info(f"Rank {rank}: Shuffled datasets for Cycle {cycle + 1}")

    train_loader = create_file_level_dataloader(train_metadata)
    val_loader = create_file_level_dataloader(val_metadata)
    test_loader = create_file_level_dataloader(test_metadata)

    highest_train_accuracy = 0
    highest_eval_accuracy = 0
    best_val_accuracy = 0
    best_model_path = f'best_model_cycle{cycle + 1}_rank{rank}.pth'

    cycle_accuracy_file = f"cycle{cycle+1}_accuracies.csv"
    with open(cycle_accuracy_file, mode='w', newline='') as cycle_file:
        cycle_writer = csv.writer(cycle_file)
        cycle_writer.writerow(["Epoch", "Training Accuracy", "Evaluation Accuracy"])

    for epoch in range(num_epochs):
        np.random.shuffle(train_metadata)
        train_loader = create_file_level_dataloader(train_metadata)

        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, train_metadata)
        logger.info(
            f"Rank {rank} - Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%"
        )

        val_loss, val_accuracy, precision, recall, f1_score = evaluate(model, val_loader, criterion, val_metadata)
        logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.2f}%")
        logger.info(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

        scheduler.step(val_loss)

        highest_train_accuracy = max(highest_train_accuracy, train_accuracy)
        highest_eval_accuracy = max(highest_eval_accuracy, val_accuracy)

        with open(cycle_accuracy_file, mode='a', newline='') as cycle_file:
            cycle_writer = csv.writer(cycle_file)
            cycle_writer.writerow([epoch + 1, f"{train_accuracy:.2f}%", f"{val_accuracy:.2f}%"])

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy

            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                logger.info(f"Deleted previous best model: {best_model_path}")

            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"Cycle {cycle + 1}: Rank {rank} saved best model at epoch {epoch + 1} "
                f"(Validation Accuracy = {val_accuracy:.2f}%)"
            )

    logger.info(f"Rank {rank} - Cycle {cycle + 1} Training Completed.")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Rank {rank}: Loaded best model {best_model_path} for testing.")
    else:
        logger.warning(f"Rank {rank}: No saved model found. Using last trained model for testing.")

    logger.info(f"Rank {rank}: Starting test evaluation for Cycle {cycle + 1}...")
    test_loss, test_accuracy, precision, recall, f1_score = test_evaluate(model, test_metadata, test_loader, cycle)
    last_test_accuracy = test_accuracy
    test_accuracies.append(test_accuracy)
    logger.info(f"Rank {rank}: Cycle {cycle + 1} Test Accuracy = {test_accuracy:.2f}%")

    test_accuracies_file = "test_accuracies.csv"
    file_exists = os.path.exists(test_accuracies_file)

    with open(test_accuracies_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Cycle", "Highest Eval Accuracy", "Test Accuracy"])
        writer.writerow([cycle + 1, f"{highest_eval_accuracy:.2f}%", f"{test_accuracy:.2f}%"])

    logger.info(f"Rank {rank}: Cycle {cycle + 1} test results written to {test_accuracies_file}.")

logger.info("Test accuracies for all cycles written to test_accuracies.csv.")

logger.info(f"Rank {rank}: Gathering test accuracies...")
global_test_accuracies = comm.gather(last_test_accuracy, root=0)

if rank == 0:
    final_test_accuracies = [acc for acc in global_test_accuracies]
    avg_test_acc = np.mean(final_test_accuracies)
    std_test_acc = np.std(final_test_accuracies)

    logger.info(f"Final Global Test Accuracy Across Ranks: {avg_test_acc:.2f}%")
    logger.info(f"Standard Deviation: {std_test_acc:.2f}%")

    with open('final_cycle_accuracies.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Avg Test Accuracy", "Std Dev"])
        writer.writerow([f"{avg_test_acc:.2f}%", f"{std_test_acc:.2f}%"])

    main_end_time = datetime.now()
    elapsed_str = str(main_end_time - main_start_time).split('.')[0]
    logger.info(f"Total Elapsed Time: {elapsed_str} (Hr:Mn:SS)")

comm.Barrier()
