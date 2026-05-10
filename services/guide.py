from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.85


@dataclass(frozen=True)
class VisualEvent:
    start: float
    end: float
    description: str
    screenshot: str | None = None
    confidence: float = 0.75


def synthesize_walkthrough(
    transcript: dict[str, Any],
    visual: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    max_steps = int(options.get("maxSteps", 12))
    segments = [_segment(item) for item in transcript.get("segments", [])]
    events = [_event(item) for item in visual.get("events", [])]

    if not segments and not events:
        return {
            "title": "Untitled Walkthrough",
            "summary": "No transcript or visual events were detected.",
            "steps": [],
            "warnings": ["The analysis did not produce usable transcript or visual events."],
        }

    timeline = events or [
        VisualEvent(
            start=segment.start,
            end=segment.end,
            description=segment.text,
            confidence=segment.confidence,
        )
        for segment in segments
    ]

    steps: list[dict[str, Any]] = []
    warnings: list[str] = []

    for event in timeline[:max_steps]:
        aligned = _overlapping_segments(event, segments)
        narration = " ".join(segment.text.strip() for segment in aligned if segment.text.strip())
        heading = _heading_from(narration or event.description)
        instruction = _instruction_from(narration, event.description)
        confidence_values = [event.confidence, *[segment.confidence for segment in aligned]]
        confidence = sum(confidence_values) / len(confidence_values)
        if confidence < 0.55:
            warnings.append(f"Review step {len(steps) + 1}; confidence is low.")

        steps.append(
            {
                "index": len(steps) + 1,
                "heading": heading,
                "instruction": instruction,
                "timestampStart": round(event.start, 2),
                "timestampEnd": round(event.end, 2),
                "screenshot": event.screenshot,
                "evidence": _evidence(narration, event.description),
                "confidence": round(max(0.0, min(confidence, 1.0)), 2),
            }
        )

    title = _heading_from(segments[0].text if segments else events[0].description)
    summary = _summary(segments, events)

    if not segments:
        warnings.append("No narration transcript was available; steps are based on visual analysis only.")
    if not events:
        warnings.append("No visual events were available; steps are based on narration only.")

    return {"title": title, "summary": summary, "steps": steps, "warnings": warnings}


def render_markdown(walkthrough: dict[str, Any]) -> str:
    lines = [f"# {walkthrough['title']}", "", walkthrough.get("summary", ""), ""]
    for step in walkthrough.get("steps", []):
        lines.extend([f"## {step['index']}. {step['heading']}", ""])
        if step.get("screenshot"):
            lines.extend([f"![{step['heading']}](assets/{step['screenshot']})", ""])
        lines.extend(
            [
                step["instruction"],
                "",
                f"Timestamp: {_timestamp(step['timestampStart'])} - {_timestamp(step['timestampEnd'])}",
                f"Confidence: {round(step['confidence'] * 100)}%",
            ]
        )
        if step.get("evidence"):
            lines.append(f"Evidence: {step['evidence']}")
        lines.append("")

    warnings = walkthrough.get("warnings", [])
    if warnings:
        lines.extend(["## Review Notes", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _segment(item: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        start=float(item.get("start", 0)),
        end=float(item.get("end", item.get("start", 0))),
        text=str(item.get("text", "")).strip(),
        confidence=float(item.get("confidence", 0.85)),
    )


def _event(item: dict[str, Any]) -> VisualEvent:
    return VisualEvent(
        start=float(item.get("start", 0)),
        end=float(item.get("end", item.get("start", 0))),
        description=str(item.get("description", "")).strip(),
        screenshot=item.get("screenshot"),
        confidence=float(item.get("confidence", 0.75)),
    )


def _overlapping_segments(event: VisualEvent, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    return [
        segment
        for segment in segments
        if segment.start <= event.end and segment.end >= event.start
    ]


def _heading_from(text: str) -> str:
    words = [word.strip(".,:;!?") for word in text.split() if word.strip(".,:;!?")]
    if not words:
        return "Review this step"
    compact = " ".join(words[:7])
    return compact[:1].upper() + compact[1:]


def _instruction_from(narration: str, description: str) -> str:
    if narration and description:
        return f"{narration.strip()} The screen shows: {description.strip()}"
    if narration:
        return narration.strip()
    return f"Follow the visible action: {description.strip()}"


def _evidence(narration: str, description: str) -> str:
    parts = []
    if narration:
        parts.append(f"narration: {narration}")
    if description:
        parts.append(f"visual: {description}")
    return "; ".join(parts)


def _summary(segments: list[TranscriptSegment], events: list[VisualEvent]) -> str:
    if segments:
        text = " ".join(segment.text for segment in segments[:3]).strip()
        return text if text else "Generated from screen recording narration and visual events."
    return events[0].description if events else "Generated from screen recording analysis."


def _timestamp(seconds: float) -> str:
    safe = max(0, int(seconds))
    return f"{safe // 60:02d}:{safe % 60:02d}"
