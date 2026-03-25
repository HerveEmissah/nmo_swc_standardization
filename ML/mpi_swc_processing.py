"""
Filename: mpi_swc_processing.py
Author: Herve Emissah
Created: 2025-4-10
Description: MPI-parallel GCN-Based Classification of neural morphology classification pipeline ran on HPC.
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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from mpi4py import MPI
from datetime import datetime
import logging
import glob
import sys
import csv

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Recursion limit to handle deep graph structures
sys.setrecursionlimit(3000)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label helpers
def label_to_int(label):
    return 0 if label == "apical" else 1 if label == "basal" else 2

def label_to_string(label):
    return ["apical", "basal", "other"][label]

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [N, F]
        # adj: [N, N]
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """
    Minimal-change REAL GCN.
    Each tree in a file is a graph node.
    Input feature per tree is still just the Sholl value.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = torch.relu(x)

        out = self.fc3(x)
        return out


def build_file_adjacency(num_nodes):
    if num_nodes <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)

    # Fully connected graph between trees in the file
    adj = torch.ones((num_nodes, num_nodes), dtype=torch.float32)

    # Add self-loops
    adj = adj + torch.eye(num_nodes, dtype=torch.float32)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj

# Training
def train(model, dataloader, optimizer, criterion, train_metadata):
    model.train()
    total_loss = 0.0
    correct_files = 0
    total_files = 0

    for features, labels, adj, file_name in dataloader:
        file_name = file_name[0] if isinstance(file_name, (list, tuple)) else file_name
        file_name = file_name.strip().lower()

        features = features.squeeze(0)   # [N, 1]
        labels = labels.squeeze(0)       # [N]
        adj = adj.squeeze(0)             # [N, N]

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
        outputs = model(features, adj)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        # Preserve original rule for minimum change:
        # trees labeled "other" remain unchanged
        predicted[labels == 2] = 2

        # Preserve original one-apical rule among non-other trees
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


# Evaluation
def evaluate(model, dataloader, criterion, eval_metadata):
    model.eval()
    total_loss = 0.0
    correct_files = 0
    total_files = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels, adj, file_name in dataloader:
            features = features.squeeze(0)   # [N, 1]
            labels = labels.squeeze(0)       # [N]
            adj = adj.squeeze(0)             # [N, N]

            outputs = model(features, adj)
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


# Test evaluation
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

    total_test_files = len(set([os.path.basename(meta["file_name"].strip()) for meta in test_metadata]))
    logger.info(f"Cycle {cycle + 1}: Total unique test files: {total_test_files}")

    with open(f"cycle{cycle+1}_neuron_test_accuracy.csv", "w", newline='') as csvfile:
        csvfile.write(
            f"{'file_name'.ljust(50)}"
            f"{'actual_tree_type'.ljust(20)}"
            f"{'predicted_tree_type'.ljust(20)}"
            f"{'tree_nodes'}\n"
        )

        with torch.no_grad():
            for features, labels, adj, file_name_batch in test_loader:
                features = features.squeeze(0)   # [N, 1]
                labels = labels.squeeze(0)       # [N]
                adj = adj.squeeze(0)             # [N, N]

                file_name_str = file_name_batch[0] if isinstance(file_name_batch, (list, tuple)) else file_name_batch
                file_metadata = next(meta for meta in test_metadata if meta["file_name"] == file_name_str)

                outputs = model(features, adj)
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

                    csvfile.write(
                        f"{file_name_str.ljust(50)}"
                        f"{actual_label.ljust(20)}"
                        f"{predicted_label.ljust(20)}"
                        f"{tree_nodes_str}\n"
                    )

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


# DataLoader
def create_file_level_dataloader(flattened_metadata, batch_size=1):
    file_features = []
    file_labels = []
    file_adjs = []
    file_names = []

    for file_data in flattened_metadata:
        file_name = file_data['file_name']
        trees = file_data['trees']

        if not trees:
            continue

        tree_features = []
        for tree in trees:
            sholl_value = tree.get("sholl_value", 0)
            tree_features.append(torch.tensor([sholl_value], dtype=torch.float32))

        tree_labels = [label_to_int(tree["tree_type"]) for tree in trees]

        file_features.append(torch.stack(tree_features))             # [N, 1]
        file_labels.append(torch.tensor(tree_labels, dtype=torch.long))
        file_adjs.append(build_file_adjacency(len(trees)))          # [N, N]
        file_names.append(file_name)

    dataset = list(zip(file_features, file_labels, file_adjs, file_names))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Sholl / graph preprocessing
def calculate_sholl_value(G, path, soma_node):
    soma_x, soma_y, soma_z = G.nodes[soma_node]["x"], G.nodes[soma_node]["y"], G.nodes[soma_node]["z"]

    distances = []
    for node in path:
        if node == soma_node:
            continue
        x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
        distance = np.linalg.norm(np.array([x, y, z]) - np.array([soma_x, soma_y, soma_z]))
        distances.append(distance)

    if not distances:
        return 0

    total_path_length = sum(distances)
    radial_step = total_path_length * 0.1 if total_path_length > 0 else 1.0

    max_distance = max(distances)
    radial_distances = np.arange(0, max_distance + radial_step, radial_step)

    sholl_value = 0
    for r in radial_distances:
        intersections = 0
        for node in path:
            if node == soma_node:
                continue
            x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
            distance = np.linalg.norm(np.array([x, y, z]) - np.array([soma_x, soma_y, soma_z]))
            if distance >= r and distance < r + radial_step:
                intersections += 1
        sholl_value += intersections

    return sholl_value


def read_swc_file(filename):
    swc_df = pd.read_csv(
        filename,
        sep=r'\s+',
        comment='#',
        header=None,
        names=['node_id', 'node_type', 'x', 'y', 'z', 'radius', 'parent'],
        encoding='ISO-8859-1'
    )
    swc_df = swc_df.astype({
        'node_id': int, 'node_type': int, 'x': float, 'y': float,
        'z': float, 'radius': float, 'parent': int
    })
    G = nx.DiGraph()
    for _, row in swc_df.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
        if row['parent'] != -1:
            G.add_edge(row['parent'], row['node_id'])
    return G


def find_branch_points_and_leaves(graph, tree_nodes):
    branch_points = []
    leaf_nodes = []

    for node in tree_nodes:
        if graph.degree(node) >= 2:
            branch_points.append(node)
        elif graph.degree(node) == 1:
            leaf_nodes.append(node)

    return branch_points, leaf_nodes


def calculate_cumulative_sholl_value(graph, tree_nodes, soma_node, target_nodes):
    cumulative_sholl_value = 0

    for target_node in target_nodes:
        path = list(nx.shortest_path(graph, source=soma_node, target=target_node))
        sholl_value = calculate_sholl_value(graph, path, soma_node)
        cumulative_sholl_value += sholl_value

    return cumulative_sholl_value


def classify_trees(graph, source_file_path):
    file_name = os.path.basename(source_file_path)
    soma_nodes = [node for node in graph.nodes if graph.nodes[node].get('node_type') == 1]
    if not soma_nodes:
        return None

    file_classification = {"file_name": file_name, "trees": []}
    tree_info = []

    logger.info(f"*********File: {file_name}")
    for soma_node in soma_nodes:
        for node in graph.successors(soma_node):
            tree_nodes = set()
            stack = [node]
            while stack:
                current_node = stack.pop()
                if current_node not in tree_nodes and graph.nodes[current_node].get('node_type') != 1:
                    tree_nodes.add(current_node)
                    stack.extend(graph.successors(current_node))

            if len(tree_nodes) > 1:
                node_types = [graph.nodes[n]["node_type"] for n in tree_nodes]
                tree_type = "apical" if 4 in node_types else "basal" if 3 in node_types else "other"

                branch_points, leaf_nodes = find_branch_points_and_leaves(graph, tree_nodes)
                target_nodes = branch_points + leaf_nodes
                cumulative_sholl_value = calculate_cumulative_sholl_value(graph, tree_nodes, soma_node, target_nodes)

                tree_info.append((tree_type, cumulative_sholl_value))

                file_classification["trees"].append({
                    "tree_nodes": sorted(tree_nodes),
                    "tree_type": tree_type,
                    "sholl_value": cumulative_sholl_value
                })

                logger.info(f"Tree Type: {tree_type}, cumulative_sholl_value: {cumulative_sholl_value}")
                logger.info("*******************")

    for apical_type, apical_sholl in tree_info:
        if apical_type == "apical":
            for basal_type, basal_sholl in tree_info:
                if basal_type == "basal" and apical_sholl < basal_sholl:
                    logger.info(
                        f"File {file_name}: Apical tree's Sholl value ({apical_sholl}) "
                        f"is less than a basal tree's Sholl value ({basal_sholl})"
                    )

    return file_classification if file_classification["trees"] else None


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


# Main
def main():
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

        logger.info(
            f"Rank 0: Data Split -> Train: {len(train_files)}, "
            f"Val: {len(val_files)}, Test: {len(test_files)}"
        )

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

    logger.info(
        f"Rank {rank}: Using Train: {len(train_metadata)}, "
        f"Val: {len(val_metadata)}, Test: {len(test_metadata)}"
    )

    num_cycles = 10
    num_epochs = 50
    test_accuracies = []

    if rank == 0:
        model_files = glob.glob('best_model_cycle*.pth')
        for model_file in model_files:
            os.remove(model_file)
            logger.info(f"Deleted old model: {model_file}")

    comm.Barrier()

    apical_weight = 3.0
    basal_weight = 1.0
    other_weight = 1.0
    class_weights = torch.tensor([apical_weight, basal_weight, other_weight])

    model = GCN(input_dim=1, hidden_dim=1024, output_dim=3)

    # Keep Adam as requested
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)

    # Keep criterion behavior close to original
    # criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = create_file_level_dataloader(train_metadata)
    val_loader = create_file_level_dataloader(val_metadata)
    test_loader = create_file_level_dataloader(test_metadata)

    # Keep OneCycleLR setup close to original
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    # scheduler = ReduceLROnPlateau(optimizer, "max", patience=5, factor=0.5, verbose=True)

    cycles_per_rank = list(range(num_cycles))[rank::size]
    logger.info(f"Rank {rank} will process cycles: {cycles_per_rank}")

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

            val_loss, val_accuracy, precision, recall, f1_score = evaluate(
                model, val_loader, criterion, val_metadata
            )
            logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.2f}%")
            logger.info(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

            # Preserve current scheduler behavior
            # For OneCycleLR, step once per epoch here to stay close to your current code
            scheduler.step()

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
            model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
            logger.info(f"Rank {rank}: Loaded best model {best_model_path} for testing.")
        else:
            logger.warning(f"Rank {rank}: No saved model found. Using last trained model for testing.")

        logger.info(f"Rank {rank}: Starting test evaluation for Cycle {cycle + 1}...")
        test_loss, test_accuracy, precision, recall, f1_score = test_evaluate(
            model, test_metadata, test_loader, cycle
        )
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
    gathered_test_acc = comm.gather(test_accuracies[-1] if len(test_accuracies) > 0 else 0.0, root=0)

    if rank == 0:
        final_test_accuracies = [acc for acc in gathered_test_acc]
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

if __name__ == "__main__":
    main()
