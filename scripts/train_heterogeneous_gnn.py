from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "processed" / "graph"
MODEL_DIR = ROOT / "models" / "gnn"
EVALUATION_DIR = ROOT / "evaluation" / "gnn"

SEED = 42
HIDDEN_DIM = 128
DROPOUT = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4
LOG_EVERY = 10

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2, "gold": 3}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def require_files() -> None:
    required = [
        GRAPH_DIR / "node_features.npz",
        GRAPH_DIR / "graph_edges.npz",
        GRAPH_DIR / "candidate_splits.npz",
        GRAPH_DIR / "candidate_ids.json",
        GRAPH_DIR / "job_ids.json",
        GRAPH_DIR / "skill_ids.json",
        GRAPH_DIR / "graph_manifest.json",
    ]
    required.extend(GRAPH_DIR / f"{split}_supervision.npz" for split in SPLIT_CODES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 8A files:\n" + "\n".join(missing))


def weighted_mean(
    source_features: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    weights: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    output = source_features.new_zeros((target_count, source_features.shape[1]))
    denominator = source_features.new_zeros((target_count, 1))
    messages = source_features[source_index] * weights.unsqueeze(1)
    output.index_add_(0, target_index, messages)
    denominator.index_add_(0, target_index, weights.unsqueeze(1))
    return output / denominator.clamp_min(1e-8)


class HeterogeneousGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.candidate_input = nn.Linear(input_dim, hidden_dim)
        self.job_input = nn.Linear(input_dim, hidden_dim)
        self.skill_input = nn.Linear(input_dim, hidden_dim)

        self.skill_self = nn.Linear(hidden_dim, hidden_dim)
        self.skill_from_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.skill_from_job = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.candidate_self = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_from_skill = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.job_self = nn.Linear(hidden_dim, hidden_dim)
        self.job_from_skill = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.skill_norm = nn.LayerNorm(hidden_dim)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.job_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        pair_dim = hidden_dim * 4 + 1
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode_nodes(
        self,
        candidate_x: torch.Tensor,
        job_x: torch.Tensor,
        skill_x: torch.Tensor,
        candidate_skill_index: torch.Tensor,
        candidate_skill_weight: torch.Tensor,
        job_skill_index: torch.Tensor,
        job_skill_weight: torch.Tensor,
        train_candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_base = F.gelu(self.candidate_input(candidate_x))
        job_base = F.gelu(self.job_input(job_x))
        skill_base = F.gelu(self.skill_input(skill_x))

        candidate_sources = candidate_skill_index[0]
        candidate_targets = candidate_skill_index[1]
        allowed_edges = train_candidate_mask[candidate_sources]

        candidate_to_skill = weighted_mean(
            candidate_base,
            candidate_sources[allowed_edges],
            candidate_targets[allowed_edges],
            candidate_skill_weight[allowed_edges],
            skill_base.shape[0],
        )
        job_to_skill = weighted_mean(
            job_base,
            job_skill_index[0],
            job_skill_index[1],
            job_skill_weight,
            skill_base.shape[0],
        )
        skill_hidden = self.skill_norm(
            F.gelu(
                self.skill_self(skill_base)
                + self.skill_from_candidate(candidate_to_skill)
                + self.skill_from_job(job_to_skill)
            )
        )
        skill_hidden = self.dropout(skill_hidden)

        skill_to_candidate = weighted_mean(
            skill_hidden,
            candidate_skill_index[1],
            candidate_skill_index[0],
            candidate_skill_weight,
            candidate_base.shape[0],
        )
        skill_to_job = weighted_mean(
            skill_hidden,
            job_skill_index[1],
            job_skill_index[0],
            job_skill_weight,
            job_base.shape[0],
        )

        candidate_hidden = self.candidate_norm(
            F.gelu(
                self.candidate_self(candidate_base)
                + self.candidate_from_skill(skill_to_candidate)
            )
        )
        job_hidden = self.job_norm(
            F.gelu(self.job_self(job_base) + self.job_from_skill(skill_to_job))
        )
        return self.dropout(candidate_hidden), self.dropout(job_hidden)

    def score_pairs(
        self,
        candidate_hidden: torch.Tensor,
        job_hidden: torch.Tensor,
        candidate_raw: torch.Tensor,
        job_raw: torch.Tensor,
        candidate_index: torch.Tensor,
        job_index: torch.Tensor,
    ) -> torch.Tensor:
        candidate_pair = candidate_hidden[candidate_index]
        job_pair = job_hidden[job_index]
        raw_cosine = (candidate_raw[candidate_index] * job_raw[job_index]).sum(dim=1, keepdim=True)
        pair_features = torch.cat(
            [
                candidate_pair,
                job_pair,
                torch.abs(candidate_pair - job_pair),
                candidate_pair * job_pair,
                raw_cosine,
            ],
            dim=1,
        )
        return self.scorer(pair_features)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        candidate_index: torch.Tensor,
        job_index: torch.Tensor,
    ) -> torch.Tensor:
        candidate_hidden, job_hidden = self.encode_nodes(
            graph["candidate_x"],
            graph["job_x"],
            graph["skill_x"],
            graph["candidate_skill_index"],
            graph["candidate_skill_weight"],
            graph["job_skill_index"],
            graph["job_skill_weight"],
            graph["train_candidate_mask"],
        )
        return self.score_pairs(
            candidate_hidden,
            job_hidden,
            graph["candidate_x"],
            graph["job_x"],
            candidate_index,
            job_index,
        )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def correlation(actual: np.ndarray, predicted: np.ndarray, method: str) -> float | None:
    if method == "spearman":
        actual = rankdata(actual)
        predicted = rankdata(predicted)
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return None
    return round(float(np.corrcoef(actual, predicted)[0, 1]), 4)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = predicted - actual
    return {
        "evaluated_pairs": int(len(actual)),
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "pearson": correlation(actual, predicted, "pearson"),
        "spearman": correlation(actual, predicted, "spearman"),
    }


def ranking_groups(candidate_index: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for row_index, candidate in enumerate(candidate_index.tolist()):
        groups[int(candidate)].append(row_index)
    return groups


def centred_ndcg(
    candidate_index: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    k: int,
) -> dict[str, Any]:
    values: list[float] = []
    for indices in ranking_groups(candidate_index).values():
        if len(indices) < 2:
            continue
        truth = actual[indices]
        relevance = truth - truth.min()
        if relevance.max() <= 0:
            continue
        prediction = predicted[indices]
        cutoff = min(k, len(indices))
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        predicted_order = np.argsort(-prediction, kind="mergesort")[:cutoff]
        ideal_order = np.argsort(-relevance, kind="mergesort")[:cutoff]
        dcg = float(np.sum(relevance[predicted_order] * discounts))
        ideal_dcg = float(np.sum(relevance[ideal_order] * discounts))
        if ideal_dcg > 0:
            values.append(dcg / ideal_dcg)
    return {
        "candidate_count": len(values),
        "mean": round(float(np.mean(values)), 4) if values else None,
        "relevance": "candidate-centred linear gain",
    }


def pairwise_accuracy(
    candidate_index: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    correct = 0.0
    comparisons = 0
    candidates = 0
    for indices in ranking_groups(candidate_index).values():
        local_comparisons = 0
        for left_position in range(len(indices)):
            left = indices[left_position]
            for right in indices[left_position + 1 :]:
                true_difference = actual[left] - actual[right]
                if true_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                comparisons += 1
                local_comparisons += 1
                if predicted_difference == 0:
                    correct += 0.5
                elif np.sign(predicted_difference) == np.sign(true_difference):
                    correct += 1.0
        if local_comparisons:
            candidates += 1
    return {
        "candidate_count": candidates,
        "pair_comparisons": comparisons,
        "accuracy": round(correct / comparisons, 4) if comparisons else None,
        "tie_handling": "Prediction ties receive 0.5 credit; target ties are excluded",
    }


def target_report(
    candidate_index: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "regression": regression_metrics(actual, predicted),
        "candidate_centred_ndcg": {
            "ndcg_at_5": centred_ndcg(candidate_index, actual, predicted, 5),
            "ndcg_at_10": centred_ndcg(candidate_index, actual, predicted, 10),
        },
        "pairwise_ranking_accuracy": pairwise_accuracy(candidate_index, actual, predicted),
        "prediction_distribution": {
            "minimum": round(float(predicted.min()), 4),
            "maximum": round(float(predicted.max()), 4),
            "mean": round(float(predicted.mean()), 4),
            "median": round(float(np.median(predicted)), 4),
        },
    }


def tensor_from_numpy(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def load_graph() -> tuple[dict[str, torch.Tensor], np.ndarray]:
    node_file = np.load(GRAPH_DIR / "node_features.npz")
    edge_file = np.load(GRAPH_DIR / "graph_edges.npz")
    split_codes = np.load(GRAPH_DIR / "candidate_splits.npz")["split_code"].astype(np.int8)

    graph = {
        "candidate_x": tensor_from_numpy(node_file["candidate"], torch.float32),
        "job_x": tensor_from_numpy(node_file["job"], torch.float32),
        "skill_x": tensor_from_numpy(node_file["skill"], torch.float32),
        "candidate_skill_index": tensor_from_numpy(
            edge_file["candidate_skill_edge_index"], torch.long
        ),
        "candidate_skill_weight": tensor_from_numpy(
            edge_file["candidate_skill_weight"], torch.float32
        ),
        "job_skill_index": tensor_from_numpy(edge_file["job_skill_edge_index"], torch.long),
        "job_skill_weight": tensor_from_numpy(edge_file["job_skill_weight"], torch.float32),
        "train_candidate_mask": tensor_from_numpy(split_codes == SPLIT_CODES["train"], torch.bool),
    }
    return graph, split_codes


def load_supervision(split: str) -> dict[str, torch.Tensor]:
    data = np.load(GRAPH_DIR / f"{split}_supervision.npz")
    return {
        "candidate_index": tensor_from_numpy(data["candidate_index"], torch.long),
        "job_index": tensor_from_numpy(data["job_index"], torch.long),
        "skills_score": tensor_from_numpy(data["skills_score"], torch.float32),
        "composite_score": tensor_from_numpy(data["composite_score"], torch.float32),
    }


def validate_visibility(
    supervision: dict[str, dict[str, torch.Tensor]], split_codes: np.ndarray
) -> None:
    for split, expected_code in SPLIT_CODES.items():
        indices = supervision[split]["candidate_index"].numpy()
        observed = set(split_codes[indices].tolist())
        if observed != {expected_code}:
            raise ValueError(
                f"{split} supervision candidates have split codes {observed}; expected only {expected_code}"
            )
    train_candidates = set(supervision["train"]["candidate_index"].tolist())
    for split in ("validation", "test", "gold"):
        other = set(supervision[split]["candidate_index"].tolist())
        overlap = train_candidates & other
        if overlap:
            raise ValueError(f"Training candidate leakage into {split}: {len(overlap)} candidates")


def standardisation(train: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.stack([train["skills_score"], train["composite_score"]], dim=1)
    mean = target.mean(dim=0)
    standard_deviation = target.std(dim=0).clamp_min(1e-6)
    return mean, standard_deviation


@torch.no_grad()
def predict(
    model: HeterogeneousGNN,
    graph: dict[str, torch.Tensor],
    split: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_standard_deviation: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    standardised = model(graph, split["candidate_index"], split["job_index"])
    return standardised * target_standard_deviation + target_mean


def validation_objective(
    prediction: torch.Tensor,
    split: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_standard_deviation: torch.Tensor,
) -> float:
    actual = torch.stack([split["skills_score"], split["composite_score"]], dim=1)
    return float(torch.mean(torch.abs((prediction - actual) / target_standard_deviation)).item())


def main() -> None:
    started = time.time()
    set_seed(SEED)
    require_files()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    graph, split_codes = load_graph()
    supervision = {split: load_supervision(split) for split in SPLIT_CODES}
    validate_visibility(supervision, split_codes)

    target_mean, target_standard_deviation = standardisation(supervision["train"])
    train_actual = torch.stack(
        [supervision["train"]["skills_score"], supervision["train"]["composite_score"]],
        dim=1,
    )
    train_standardised = (train_actual - target_mean) / target_standard_deviation

    graph = {key: value.to(device) for key, value in graph.items()}
    supervision = {
        split: {key: value.to(device) for key, value in values.items()}
        for split, values in supervision.items()
    }
    target_mean = target_mean.to(device)
    target_standard_deviation = target_standard_deviation.to(device)
    train_standardised = train_standardised.to(device)

    model = HeterogeneousGNN(
        input_dim=int(graph["candidate_x"].shape[1]),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(f"Training on {device.type.upper()} with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Gold labels are locked until training and model selection finish.\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_prediction_standardised = model(
            graph,
            supervision["train"]["candidate_index"],
            supervision["train"]["job_index"],
        )
        train_loss = F.smooth_l1_loss(train_prediction_standardised, train_standardised)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        validation_prediction = predict(
            model,
            graph,
            supervision["validation"],
            target_mean,
            target_standard_deviation,
        )
        validation_score = validation_objective(
            validation_prediction,
            supervision["validation"],
            target_mean,
            target_standard_deviation,
        )
        history.append(
            {
                "epoch": epoch,
                "train_smooth_l1": round(float(train_loss.item()), 6),
                "validation_standardised_mae": round(validation_score, 6),
            }
        )

        if validation_score < best_validation - MIN_DELTA:
            best_validation = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss.item():.5f} "
                f"| validation standardised MAE {validation_score:.5f}"
            )
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid model checkpoint")
    model.load_state_dict(best_state)

    # Gold is touched only here, after the best epoch has been selected using validation.
    candidate_ids = [str(value) for value in load_json(GRAPH_DIR / "candidate_ids.json")]
    job_ids = [str(value) for value in load_json(GRAPH_DIR / "job_ids.json")]
    report_splits: dict[str, Any] = {}
    for split, values in supervision.items():
        prediction = predict(model, graph, values, target_mean, target_standard_deviation).cpu().numpy()
        candidate_index = values["candidate_index"].cpu().numpy()
        job_index = values["job_index"].cpu().numpy()
        skills_actual = values["skills_score"].cpu().numpy()
        composite_actual = values["composite_score"].cpu().numpy()

        report_splits[split] = {
            "pair_count": int(len(candidate_index)),
            "candidate_count": int(len(np.unique(candidate_index))),
            "skills_target": target_report(candidate_index, skills_actual, prediction[:, 0]),
            "composite_target": target_report(
                candidate_index, composite_actual, prediction[:, 1]
            ),
        }

        prediction_rows = [
            {
                "candidate_id": candidate_ids[int(candidate)],
                "job_id": job_ids[int(job)],
                "skills_actual": round(float(skills_actual[row]), 6),
                "skills_predicted": round(float(prediction[row, 0]), 6),
                "composite_actual": round(float(composite_actual[row]), 6),
                "composite_predicted": round(float(prediction[row, 1]), 6),
            }
            for row, (candidate, job) in enumerate(zip(candidate_index, job_index))
        ]
        save_json(EVALUATION_DIR / f"{split}_predictions.json", prediction_rows)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(graph["candidate_x"].shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "target_mean": target_mean.cpu(),
        "target_standard_deviation": target_standard_deviation.cpu(),
        "best_epoch": best_epoch,
        "seed": SEED,
    }
    torch.save(checkpoint, MODEL_DIR / "heterogeneous_gnn.pt")

    elapsed = time.time() - started
    report = {
        "model": {
            "name": "Leakage-safe weighted heterogeneous GNN",
            "framework": f"PyTorch {torch.__version__}",
            "device": device.type,
            "input_dimension": int(graph["candidate_x"].shape[1]),
            "hidden_dimension": HIDDEN_DIM,
            "dropout": DROPOUT,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "message_passing": [
                "training-candidate to skill",
                "all-job to skill",
                "skill to candidate",
                "skill to job",
            ],
        },
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "loss": "Equal-weight Smooth L1 on train-standardised skills and composite targets",
            "selection_metric": "Validation standardised mean absolute error across both targets",
            "best_epoch": best_epoch,
            "best_validation_standardised_mae": round(best_validation, 6),
            "epochs_completed": len(history),
            "elapsed_seconds": round(elapsed, 2),
            "target_mean_from_train": [round(float(x), 6) for x in target_mean.cpu()],
            "target_standard_deviation_from_train": [
                round(float(x), 6) for x in target_standard_deviation.cpu()
            ],
            "history": history,
        },
        "splits": report_splits,
        "safety_rules": {
            "training_candidate_nodes_only_update_skill_nodes": True,
            "validation_candidates_used_for_gradient_updates": False,
            "test_candidates_used_for_gradient_updates": False,
            "gold_candidates_used_for_gradient_updates": False,
            "validation_used_for_early_stopping": True,
            "test_used_for_model_selection": False,
            "gold_used_for_model_selection": False,
            "candidate_job_labels_used_as_message_edges": False,
            "all_job_features_available_under_candidate_cold_start_protocol": True,
            "prediction_scores_artificially_offset_or_capped": False,
        },
        "files": {
            "checkpoint": str((MODEL_DIR / "heterogeneous_gnn.pt").relative_to(ROOT)),
            "report": str((EVALUATION_DIR / "gnn_report.json").relative_to(ROOT)),
            "predictions": {
                split: str((EVALUATION_DIR / f"{split}_predictions.json").relative_to(ROOT))
                for split in SPLIT_CODES
            },
        },
    }
    save_json(EVALUATION_DIR / "gnn_report.json", report)

    test_skills = report_splits["test"]["skills_target"]["regression"]
    test_composite = report_splits["test"]["composite_target"]["regression"]
    print("\nStage 8B completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Test skills MAE: {test_skills['mae']:.4f}")
    print(f"Test composite MAE: {test_composite['mae']:.4f}")
    print(f"Model: {MODEL_DIR / 'heterogeneous_gnn.pt'}")
    print(f"Report: {EVALUATION_DIR / 'gnn_report.json'}")


if __name__ == "__main__":
    main()
