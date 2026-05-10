import { describe, expect, it } from "vitest";
import { buildMarkdown, formatTimestamp, synthesizeWalkthrough, type Walkthrough } from "./guide";

describe("guide markdown", () => {
  it("renders tutorial steps with screenshots and confidence", () => {
    const walkthrough: Walkthrough = {
      title: "Settings walkthrough",
      summary: "Configure the Modal worker.",
      steps: [
        {
          index: 1,
          heading: "Open settings",
          instruction: "Open the settings panel and paste the Modal URL.",
          timestampStart: 0,
          timestampEnd: 4,
          screenshot: "step-001.jpg",
          evidence: "narration and visual panel state",
          confidence: 0.84
        }
      ],
      warnings: ["Review token handling before sharing."]
    };

    const markdown = buildMarkdown(walkthrough);

    expect(markdown).toContain("# Settings walkthrough");
    expect(markdown).toContain("![Open settings](assets/step-001.jpg)");
    expect(markdown).toContain("Confidence: 84%");
    expect(markdown).toContain("## Review Notes");
  });

  it("formats timestamps defensively", () => {
    expect(formatTimestamp(65.9)).toBe("01:05");
    expect(formatTimestamp(-5)).toBe("00:00");
  });

  it("synthesizes steps locally from separate Whisper and visual VM responses", () => {
    const walkthrough = synthesizeWalkthrough(
      {
        segments: [{ start: 0, end: 4, text: "Paste the Whisper ASR endpoint.", confidence: 0.9 }]
      },
      {
        events: [
          {
            start: 1,
            end: 5,
            description: "The settings form shows separate SmolVLM and Whisper ASR URL fields.",
            screenshot: "step-001.jpg",
            confidence: 0.8
          }
        ]
      }
    );

    expect(walkthrough.steps).toHaveLength(1);
    expect(walkthrough.steps[0].instruction).toContain("Paste the Whisper ASR endpoint.");
    expect(walkthrough.steps[0].evidence).toContain("SmolVLM");
  });
});
