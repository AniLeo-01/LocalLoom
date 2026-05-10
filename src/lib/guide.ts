export type WalkthroughStep = {
  index: number;
  heading: string;
  instruction: string;
  timestampStart: number;
  timestampEnd: number;
  screenshot?: string;
  evidence: string;
  confidence: number;
};

export type Walkthrough = {
  title: string;
  summary: string;
  steps: WalkthroughStep[];
  warnings: string[];
};

export type TranscriptResponse = {
  segments: Array<{ start: number; end: number; text: string; confidence: number }>;
  duration?: number;
};

export type VisualResponse = {
  events: Array<{
    start: number;
    end: number;
    description: string;
    screenshot?: string;
    screenshotData?: string;
    confidence: number;
  }>;
};

export function synthesizeWalkthrough(
  transcript: TranscriptResponse,
  visual: VisualResponse,
  options: { maxSteps?: number } = {}
): Walkthrough {
  const maxSteps = options.maxSteps ?? 12;
  const segments = transcript.segments ?? [];
  const events = visual.events ?? [];

  if (segments.length === 0 && events.length === 0) {
    return {
      title: "Untitled Walkthrough",
      summary: "No transcript or visual events were detected.",
      steps: [],
      warnings: ["The analysis did not produce usable transcript or visual events."]
    };
  }

  const timeline: VisualResponse["events"] =
    events.length > 0
      ? events
      : segments.map((segment) => ({
          start: segment.start,
          end: segment.end,
          description: segment.text,
          confidence: segment.confidence
        }));

  const warnings: string[] = [];
  const steps: WalkthroughStep[] = timeline.slice(0, maxSteps).map((event, index) => {
    const aligned = segments.filter((segment) => segment.start <= event.end && segment.end >= event.start);
    const narration = aligned.map((segment) => segment.text.trim()).filter(Boolean).join(" ");
    const confidenceValues = [event.confidence, ...aligned.map((segment) => segment.confidence)];
    const confidence = confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length;
    if (confidence < 0.55) warnings.push(`Review step ${index + 1}; confidence is low.`);

    return {
      index: index + 1,
      heading: headingFrom(narration || event.description),
      instruction: instructionFrom(narration, event.description),
      timestampStart: roundTime(event.start),
      timestampEnd: roundTime(event.end),
      screenshot: event.screenshot,
      evidence: evidenceFrom(narration, event.description),
      confidence: Math.max(0, Math.min(roundTime(confidence), 1))
    };
  });

  if (segments.length === 0) warnings.push("No narration transcript was available; steps are based on visual analysis only.");
  if (events.length === 0) warnings.push("No visual events were available; steps are based on narration only.");

  return {
    title: headingFrom(segments[0]?.text || events[0]?.description || "Untitled Walkthrough"),
    summary: summaryFrom(segments, events),
    steps,
    warnings
  };
}

export function buildMarkdown(walkthrough: Walkthrough): string {
  const lines: string[] = [`# ${walkthrough.title}`, "", walkthrough.summary, ""];

  for (const step of walkthrough.steps) {
    lines.push(`## ${step.index}. ${step.heading}`);
    lines.push("");
    if (step.screenshot) {
      lines.push(`![${step.heading}](assets/${step.screenshot})`);
      lines.push("");
    }
    lines.push(step.instruction);
    lines.push("");
    lines.push(`Timestamp: ${formatTimestamp(step.timestampStart)} - ${formatTimestamp(step.timestampEnd)}`);
    lines.push(`Confidence: ${Math.round(step.confidence * 100)}%`);
    if (step.evidence) {
      lines.push(`Evidence: ${step.evidence}`);
    }
    lines.push("");
  }

  if (walkthrough.warnings.length > 0) {
    lines.push("## Review Notes");
    lines.push("");
    walkthrough.warnings.forEach((warning) => lines.push(`- ${warning}`));
    lines.push("");
  }

  return lines.join("\n").trimEnd() + "\n";
}

export function formatTimestamp(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60).toString().padStart(2, "0");
  const rest = (safeSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function headingFrom(text: string): string {
  const words = text
    .split(/\s+/)
    .map((word) => word.replace(/[.,:;!?]+$/g, ""))
    .filter(Boolean);
  if (words.length === 0) return "Review this step";
  const compact = words.slice(0, 7).join(" ");
  return compact.slice(0, 1).toUpperCase() + compact.slice(1);
}

function instructionFrom(narration: string, description: string): string {
  if (narration && description) return `${narration.trim()} The screen shows: ${description.trim()}`;
  if (narration) return narration.trim();
  return `Follow the visible action: ${description.trim()}`;
}

function evidenceFrom(narration: string, description: string): string {
  return [
    narration ? `narration: ${narration}` : "",
    description ? `visual: ${description}` : ""
  ]
    .filter(Boolean)
    .join("; ");
}

function summaryFrom(
  segments: TranscriptResponse["segments"],
  events: VisualResponse["events"]
): string {
  if (segments.length > 0) {
    return segments
      .slice(0, 3)
      .map((segment) => segment.text)
      .join(" ")
      .trim() || "Generated from screen recording narration and visual events.";
  }
  return events[0]?.description || "Generated from screen recording analysis.";
}

function roundTime(value: number): number {
  return Math.round(value * 100) / 100;
}
