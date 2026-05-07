#!/usr/bin/env python3
"""
Harness evaluation script — runs inside the sandbox after training.

Usage:
    python -m harness.evaluate \
        --criteria criteria.yaml \
        --results results.json \
        --experiment-id 042 \
        [--proposal-id 042-monarch-gated] \
        [--report-url http://api:8000/experiments/042/verdict]

Reads criteria.yaml and results.json, compares metrics against thresholds,
optionally checks that the agent only edited allowed paths (via git diff),
and writes verdict.json.

Exit code:
    0  — all required criteria passed
    1  — at least one required criterion failed (or error)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from harness.schema import (
    Comparator,
    Criterion,
    CriterionResult,
    EvaluationVerdict,
    ExperimentCriteria,
    PathViolation,
    TaskCriteria,
)


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------


def evaluate_criterion(criterion: Criterion, metrics: dict) -> CriterionResult:
    """Compare a single metric against its criterion."""
    value = _resolve_metric(criterion.metric_key, metrics)

    if value is None:
        return CriterionResult(
            name=criterion.name,
            metric_key=criterion.metric_key,
            target=criterion.target,
            achieved=None,
            comparator=criterion.comparator,
            passed=False,
            required=criterion.required,
            detail=f"Metric '{criterion.metric_key}' not found in results",
        )

    passed = _compare(value, criterion.target, criterion.comparator, criterion.target_upper)

    return CriterionResult(
        name=criterion.name,
        metric_key=criterion.metric_key,
        target=criterion.target,
        achieved=value,
        comparator=criterion.comparator,
        passed=passed,
        required=criterion.required,
    )


def _resolve_metric(key: str, metrics: dict) -> Optional[float]:
    """Resolve a dot-path or slash-path key like 's5/val_acc' from a nested dict."""
    # Try slash-separated, then dot-separated
    for sep in ("/", "."):
        parts = key.split(sep)
        obj = metrics
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                obj = None
                break
        if obj is not None and not isinstance(obj, dict):
            try:
                return float(obj)
            except (TypeError, ValueError):
                return None
    # Also try as a flat key
    if key in metrics:
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            pass
    return None


def _compare(
    value: float,
    target: float,
    comparator: Comparator,
    target_upper: Optional[float] = None,
) -> bool:
    match comparator:
        case Comparator.GTE:
            return value >= target
        case Comparator.LTE:
            return value <= target
        case Comparator.GT:
            return value > target
        case Comparator.LT:
            return value < target
        case Comparator.EQ:
            return abs(value - target) < 1e-9
        case Comparator.BETWEEN:
            upper = target_upper if target_upper is not None else target
            return target <= value <= upper
    return False


# ---------------------------------------------------------------------------
# Path violation checks
# ---------------------------------------------------------------------------


def check_path_violations(
    allowed: list[str], forbidden: list[str]
) -> list[PathViolation]:
    """Use git diff to find files the agent modified outside allowed paths."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # If HEAD~0 doesn't show changes (no commits), try diff against initial state
        if not result.stdout.strip():
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        changed = [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []  # Can't check — don't fail for this

    violations = []
    for filepath in changed:
        in_allowed = any(filepath.startswith(a) for a in allowed)
        in_forbidden = any(filepath.startswith(f) for f in forbidden)
        if in_forbidden or not in_allowed:
            violations.append(PathViolation(path=filepath))
    return violations


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------


def load_results(results_path: Path) -> dict:
    """Load metrics from results.json."""
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return json.load(f)


def load_criteria(criteria_path: Path) -> ExperimentCriteria:
    """Load criteria from YAML file."""
    with open(criteria_path) as f:
        raw = yaml.safe_load(f)
    return ExperimentCriteria(**raw)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    criteria: ExperimentCriteria,
    metrics: dict,
    experiment_id: str,
    proposal_id: str,
    wall_time_seconds: Optional[float] = None,
) -> EvaluationVerdict:
    """Run all criteria checks and produce a verdict."""
    results: list[CriterionResult] = []

    for task in criteria.tasks:
        for criterion in task.criteria:
            result = evaluate_criterion(criterion, metrics)
            results.append(result)

    path_violations = check_path_violations(
        criteria.allowed_edit_paths, criteria.forbidden_edit_paths
    )

    training_completed = len(metrics) > 0
    required_passed = all(r.passed for r in results if r.required)
    no_violations = len(path_violations) == 0
    overall_pass = required_passed and no_violations and training_completed

    return EvaluationVerdict(
        experiment_id=experiment_id,
        proposal_id=proposal_id,
        overall_pass=overall_pass,
        criteria_results=results,
        training_completed=training_completed,
        wall_time_seconds=wall_time_seconds,
        path_violations=path_violations,
        raw_metrics=metrics,
    )


def print_verdict(verdict: EvaluationVerdict) -> None:
    """Print a human-readable verdict table."""
    print("\n" + "=" * 70)
    print(f"  EVALUATION VERDICT — {'PASS' if verdict.overall_pass else 'FAIL'}")
    print("=" * 70)
    print(f"  Experiment: {verdict.experiment_id}")
    print(f"  Proposal:   {verdict.proposal_id}")
    if verdict.wall_time_seconds is not None:
        print(f"  Wall time:  {verdict.wall_time_seconds:.0f}s")
    print()

    # Criteria table
    print(f"  {'Criterion':<30} {'Target':>10} {'Achieved':>10} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for r in verdict.criteria_results:
        achieved_str = f"{r.achieved:.4f}" if r.achieved is not None else "N/A"
        status = "PASS" if r.passed else "FAIL"
        req = "*" if r.required else " "
        print(f"  {r.name:<30} {r.target:>10.4f} {achieved_str:>10} {status:>6}{req}")

    if verdict.path_violations:
        print(f"\n  Path violations ({len(verdict.path_violations)}):")
        for v in verdict.path_violations:
            print(f"    - {v.path} ({v.action})")

    print()
    print(f"  * = required criterion")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment against criteria")
    parser.add_argument(
        "--criteria", type=Path, required=True, help="Path to criteria.yaml"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results.json"), help="Path to results.json"
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    parser.add_argument("--proposal-id", default="", help="Proposal ID")
    parser.add_argument(
        "--report-url",
        default=None,
        help="URL to POST verdict to (e.g. http://api/experiments/042/verdict)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("verdict.json"),
        help="Path to write verdict.json",
    )
    args = parser.parse_args()

    criteria = load_criteria(args.criteria)
    metrics = load_results(args.results)

    verdict = run_evaluation(
        criteria=criteria,
        metrics=metrics,
        experiment_id=args.experiment_id,
        proposal_id=args.proposal_id,
    )

    # Write verdict
    verdict_dict = verdict.model_dump()
    with open(args.output, "w") as f:
        json.dump(verdict_dict, f, indent=2)

    print_verdict(verdict)

    # Optionally report to API
    if args.report_url:
        try:
            import urllib.request

            req = urllib.request.Request(
                args.report_url,
                data=json.dumps(verdict_dict).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            print(f"  Verdict reported to {args.report_url}")
        except Exception as e:
            print(f"  Warning: failed to report verdict: {e}", file=sys.stderr)

    sys.exit(0 if verdict.overall_pass else 1)


if __name__ == "__main__":
    main()
