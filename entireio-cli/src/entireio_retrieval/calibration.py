from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .models import Answerability, QueryRecord, RelevanceJudgment, RetrievalHit

CALIBRATION_VERSION = "score-calibration-v1"
ANSWERABILITY_FEATURES = [
    "max_dense_score",
    "top_1_2_margin",
    "top_1_10_drop",
    "bm25_overlap_at_10",
    "calibrated_relevant_count_at_10",
    "mean_top_5_relevance_probability",
]


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    reliability = []
    expected_calibration_error = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if not np.any(mask):
            continue
        mean_probability = float(np.mean(probabilities[mask]))
        observed_rate = float(np.mean(labels[mask]))
        count = int(np.sum(mask))
        expected_calibration_error += (
            count / len(labels) * abs(mean_probability - observed_rate)
        )
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
            }
        )
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(expected_calibration_error),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "reliability_bins": reliability,
    }


def _best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str,
    minimum_recall: float = 0.0,
) -> float:
    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        if objective == "f1":
            precision = precision_score(labels, predicted, zero_division=0)
            recall = recall_score(labels, predicted, zero_division=0)
            value = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        else:
            value = balanced_accuracy_score(labels, predicted)
        if recall_score(labels, predicted, zero_division=0) < minimum_recall:
            continue
        candidate = (float(value), float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("cannot choose a threshold from an empty score set")
    return best[1]


def relevance_probability(score: float, artifact: dict[str, Any]) -> float:
    calibration = artifact["relevance"]
    return float(
        np.interp(
            score,
            calibration["x_thresholds"],
            calibration["y_thresholds"],
            left=calibration["y_thresholds"][0],
            right=calibration["y_thresholds"][-1],
        )
    )


def query_features(
    dense_scores: list[float],
    dense_ids: list[str],
    bm25_ids: list[str],
    artifact: dict[str, Any],
) -> list[float]:
    if not dense_scores:
        return [0.0] * len(ANSWERABILITY_FEATURES)
    padded = dense_scores + [dense_scores[-1]] * max(0, 10 - len(dense_scores))
    dense_top10 = set(dense_ids[:10])
    lexical_top10 = set(bm25_ids[:10])
    probabilities = [
        relevance_probability(score, artifact) for score in padded[:10]
    ]
    return [
        dense_scores[0],
        dense_scores[0] - padded[1],
        dense_scores[0] - padded[9],
        len(dense_top10 & lexical_top10) / 10,
        float(sum(probability >= artifact["relevance"]["probability_threshold"] for probability in probabilities)),
        float(np.mean(probabilities[:5])),
    ]


def answerability_probability(features: list[float], artifact: dict[str, Any]) -> float:
    model = artifact["answerability"]
    values = np.asarray(features, dtype=np.float64)
    scaled = (values - np.asarray(model["feature_mean"])) / np.asarray(
        model["feature_scale"]
    )
    logit = float(
        np.asarray(model["coefficients"]) @ scaled + model["intercept"]
    )
    return float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))


def annotate_and_filter_hits(
    hits: list[RetrievalHit],
    artifact: dict[str, Any],
) -> list[RetrievalHit]:
    output = []
    threshold = artifact["relevance"]["raw_score_threshold"]
    filtering_enabled = artifact["relevance"].get(
        "enabled_for_filtering", False
    )
    for hit in hits:
        probability = (
            relevance_probability(hit.dense_score, artifact)
            if hit.dense_score is not None
            else 0.0
        )
        lexical_rescue = hit.lexical_rank is not None and hit.lexical_rank <= 5
        accepted = (
            hit.dense_score is not None and hit.dense_score >= threshold
        ) or lexical_rescue
        annotated = hit.model_copy(
            update={
                "payload": {
                    **hit.payload,
                    "relevance_probability": probability,
                    "calibration_version": artifact["version"],
                    "passes_relevance_threshold": accepted,
                    "calibration_filtering_enabled": filtering_enabled,
                    "lexical_rescue": lexical_rescue,
                }
            }
        )
        if accepted or not filtering_enabled:
            output.append(annotated)
    return output


def _trace_rows(
    traces: dict,
    judgments: list[RelevanceJudgment],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    qrels = {
        (judgment.query_id, judgment.evidence_id): judgment.grade
        for judgment in judgments
    }
    scores, labels, groups = [], [], []
    for query_id, rows in traces.items():
        for row in rows:
            grade = qrels.get((query_id, row["evidence_id"]))
            score = row.get("dense_score")
            if grade is None or score is None:
                continue
            scores.append(float(score))
            labels.append(int(grade > 0))
            groups.append(query_id)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        groups,
    )


def _query_matrix(
    dense_traces: dict,
    bm25_traces: dict,
    queries: list[QueryRecord],
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, labels, query_ids = [], [], []
    for query in queries:
        dense_rows = dense_traces.get(query.query_id, [])
        lexical_rows = bm25_traces.get(query.query_id, [])
        dense_scores = [
            float(row["dense_score"])
            for row in dense_rows
            if row.get("dense_score") is not None
        ]
        dense_ids = [row["evidence_id"] for row in dense_rows]
        bm25_ids = [row["evidence_id"] for row in lexical_rows]
        features.append(query_features(dense_scores, dense_ids, bm25_ids, artifact))
        labels.append(int(query.answerability != Answerability.UNSUPPORTED))
        query_ids.append(query.query_id)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        query_ids,
    )


def fit_calibration(
    validation_traces: dict,
    validation_queries: list[QueryRecord],
    judgments: list[RelevanceJudgment],
    *,
    evaluation_traces: dict | None = None,
    evaluation_queries: list[QueryRecord] | None = None,
) -> dict[str, Any]:
    scores, labels, groups = _trace_rows(validation_traces["dense"], judgments)
    if len(np.unique(labels)) != 2:
        raise ValueError("relevance calibration requires relevant and irrelevant labels")
    raw_threshold = _best_threshold(labels, scores, objective="f1")
    grouped_oof = np.zeros_like(scores)
    group_values = np.asarray(groups)
    group_folds = GroupKFold(n_splits=min(5, len(set(groups))))
    for train_indices, test_indices in group_folds.split(
        scores, labels, groups=group_values
    ):
        fold_model = IsotonicRegression(out_of_bounds="clip")
        fold_model.fit(scores[train_indices], labels[train_indices])
        grouped_oof[test_indices] = fold_model.predict(scores[test_indices])
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(scores, labels)
    probabilities = isotonic.predict(scores)
    probability_threshold = relevance_probability(
        raw_threshold,
        {
            "relevance": {
                "x_thresholds": isotonic.X_thresholds_.tolist(),
                "y_thresholds": isotonic.y_thresholds_.tolist(),
            }
        },
    )
    artifact: dict[str, Any] = {
        "version": CALIBRATION_VERSION,
        "trained_partition": "validation",
        "model_variant": "pretrained_dense",
        "relevance": {
            "method": "isotonic",
            "raw_score_threshold": raw_threshold,
            "probability_threshold": probability_threshold,
            "x_thresholds": isotonic.X_thresholds_.tolist(),
            "y_thresholds": isotonic.y_thresholds_.tolist(),
            "enabled_for_filtering": False,
            "validation_oof_metrics": _binary_metrics(
                labels, grouped_oof, probability_threshold
            ),
            "validation_fit_metrics": _binary_metrics(
                labels, probabilities, probability_threshold
            ),
            "judged_pairs": len(scores),
            "query_groups": len(set(groups)),
        },
    }

    matrix, answerable, query_ids = _query_matrix(
        validation_traces["dense"],
        validation_traces["bm25"],
        validation_queries,
        artifact,
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=20260729,
    )
    negative_count = int(sum(answerable == 0))
    folds = min(5, negative_count, int(sum(answerable == 1)))
    if folds < 2:
        raise ValueError("answerability calibration requires at least two examples per class")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260729)
    oof = cross_val_predict(
        model,
        scaled,
        answerable,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    answerability_threshold = _best_threshold(
        answerable,
        oof,
        objective="balanced_accuracy",
        minimum_recall=0.8,
    )
    validation_metrics = _binary_metrics(
        answerable, oof, answerability_threshold
    )
    enabled_for_abstention = (
        validation_metrics["roc_auc"] >= 0.65
        and validation_metrics["balanced_accuracy"] >= 0.60
        and validation_metrics["recall"] >= 0.80
    )
    model.fit(scaled, answerable)
    artifact["answerability"] = {
        "method": "balanced_logistic_regression",
        "features": ANSWERABILITY_FEATURES,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "probability_threshold": answerability_threshold,
        "enabled_for_abstention": enabled_for_abstention,
        "quality_gate": {
            "minimum_roc_auc": 0.65,
            "minimum_balanced_accuracy": 0.60,
            "minimum_answerable_recall": 0.80,
        },
        "validation_oof_metrics": validation_metrics,
        "validation_queries": len(query_ids),
        "unsupported_validation_queries": negative_count,
    }

    if evaluation_traces is not None and evaluation_queries is not None:
        eval_scores, eval_labels, _ = _trace_rows(
            evaluation_traces["dense"], judgments
        )
        eval_probabilities = isotonic.predict(eval_scores)
        eval_matrix, eval_answerable, _ = _query_matrix(
            evaluation_traces["dense"],
            evaluation_traces["bm25"],
            evaluation_queries,
            artifact,
        )
        eval_scaled = (eval_matrix - scaler.mean_) / scaler.scale_
        eval_answerability = model.predict_proba(eval_scaled)[:, 1]
        artifact["holdout_evaluation"] = {
            "relevance_metrics": _binary_metrics(
                eval_labels, eval_probabilities, probability_threshold
            ),
            "answerability_metrics": _binary_metrics(
                eval_answerable,
                eval_answerability,
                answerability_threshold,
            ),
            "judged_pairs": len(eval_scores),
            "queries": len(evaluation_queries),
        }
    return artifact
