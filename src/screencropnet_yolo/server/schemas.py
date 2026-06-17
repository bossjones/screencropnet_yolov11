"""Pydantic v2 DTOs shared across the API, worker, client, and export layers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClassifyAccepted(BaseModel):
    job_id: str
    batch_id: str


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    batch_id: str
    original_path: str
    status: str
    is_twitter: bool | None = None
    pred_class: str | None = None
    pred_prob: float | None = None
    time_for_pred: float | None = None
    error: str | None = None


class StatusSummary(BaseModel):
    batch_id: str | None
    total: int
    counts: dict[str, int]
    twitter_count: int
    done: int
    failed: int
    throughput_per_sec: float


class QueueMessage(BaseModel):
    job_id: str
    batch_id: str
    compressed_path: str
    original_path: str


class ExportRecord(BaseModel):
    original_path: str
    dest_path: str
    index: int
    copied: bool
    reason: str
