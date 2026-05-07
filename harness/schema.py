"""
Harness schema — Pydantic models for experiment criteria and evaluation verdicts.

These models define the contract between:
  1. Proposal authors (who specify success criteria)
  2. The agent (who sees criteria in its prompt)
  3. The evaluation script (who checks results against criteria)
  4. The API (who stores and displays verdicts)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Criteria (input side — defined before the agent runs)
# ---------------------------------------------------------------------------


class Comparator(str, Enum):
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="
    BETWEEN = "between"


class Criterion(BaseModel):
    """One row of the evaluation table (cf. Table 1 in the paper)."""

    name: str = Field(..., description="Human-readable criterion name")
    metric_key: str = Field(
        ...,
        description="Dot-path key to look up in results, e.g. 's5/val_acc'",
    )
    target: float = Field(..., description="Threshold value")
    comparator: Comparator = Field(
        Comparator.GTE, description="How to compare achieved vs target"
    )
    target_upper: Optional[float] = Field(
        None, description="Upper bound (only for 'between' comparator)"
    )
    required: bool = Field(
        True, description="If True, failure on this criterion => experiment fails"
    )
    description: str = Field("", description="Human-readable explanation")


class TaskCriteria(BaseModel):
    """Criteria grouped by task (e.g., S5, addition)."""

    task_name: str
    criteria: list[Criterion]


class ExperimentCriteria(BaseModel):
    """Full criteria contract for an experiment — attached to a proposal."""

    tasks: list[TaskCriteria]
    time_budget_minutes: int = Field(
        30, description="Wall-clock time limit for the experiment"
    )
    baseline_config: Optional[str] = Field(
        None, description="Path to baseline YAML config for comparison"
    )
    allowed_edit_paths: list[str] = Field(
        default=["models/", "configs/"],
        description="Filesystem paths the agent may create/modify",
    )
    forbidden_edit_paths: list[str] = Field(
        default=["train/", "tasks/", "harness/", "evaluate.py"],
        description="Filesystem paths the agent must not touch",
    )


# ---------------------------------------------------------------------------
# Verdict (output side — produced after the agent runs)
# ---------------------------------------------------------------------------


class CriterionResult(BaseModel):
    """One row of the verdict table."""

    name: str
    metric_key: str
    target: float
    achieved: Optional[float] = None
    comparator: Comparator
    passed: bool
    required: bool
    detail: str = ""


class PathViolation(BaseModel):
    """A file the agent edited outside allowed paths."""

    path: str
    action: str = "modified"


class EvaluationVerdict(BaseModel):
    """Structured output of evaluate.py — stored in experiments.results."""

    experiment_id: str
    proposal_id: str
    overall_pass: bool
    criteria_results: list[CriterionResult]
    training_completed: bool
    wall_time_seconds: Optional[float] = None
    path_violations: list[PathViolation] = Field(default_factory=list)
    error: Optional[str] = None
    raw_metrics: dict = Field(
        default_factory=dict, description="Full metrics dump for archival"
    )
