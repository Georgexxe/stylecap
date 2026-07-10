"""Validated data contracts for perception, generation, and caption selection."""

from pydantic import BaseModel, Field


class HumorHook(BaseModel):
    """A grounded observation and its potential use in a humorous caption."""

    observation: str
    why_funny: str


class TimelineBeat(BaseModel):
    t: str  # "0:12"
    event: str


class FactSheet(BaseModel):
    clip_id: str
    setting: str
    entities: list[str]
    actions: list[str]
    timeline: list[TimelineBeat]
    on_screen_text: list[str] = Field(default_factory=list)
    audio_events: list[str] = Field(default_factory=list)
    mood: str
    humor_hooks: list[HumorHook] = Field(default_factory=list)


class Candidate(BaseModel):
    clip_id: str
    style: str
    caption: str
    # Optional evaluation metadata.
    accuracy_pass: bool | None = None
    unverified_claims: list[str] = Field(default_factory=list)
    tone_scores: dict[str, float] = Field(default_factory=dict)
    mean_score: float | None = None
    critique: str | None = None


class FinalCaption(BaseModel):
    clip_id: str
    style: str
    caption: str
