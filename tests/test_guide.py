import unittest

from services.guide import render_markdown, synthesize_walkthrough


class GuideSynthesisTest(unittest.TestCase):
    def test_aligns_transcript_with_visual_events(self):
        walkthrough = synthesize_walkthrough(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "Open settings and paste the Modal URL.",
                        "confidence": 0.9,
                    }
                ]
            },
            {
                "events": [
                    {
                        "start": 1,
                        "end": 5,
                        "description": "Settings panel with Modal URL field.",
                        "screenshot": "step-001.jpg",
                        "confidence": 0.8,
                    }
                ]
            },
            {"maxSteps": 4},
        )

        self.assertEqual(len(walkthrough["steps"]), 1)
        self.assertIn("Open settings", walkthrough["steps"][0]["instruction"])
        self.assertIn("Settings panel", walkthrough["steps"][0]["evidence"])

    def test_visual_only_adds_warning(self):
        walkthrough = synthesize_walkthrough(
            {"segments": []},
            {"events": [{"start": 0, "end": 2, "description": "A save button is clicked."}]},
        )

        self.assertEqual(len(walkthrough["steps"]), 1)
        self.assertTrue(any("No narration" in warning for warning in walkthrough["warnings"]))

    def test_markdown_renders_assets_and_review_notes(self):
        markdown = render_markdown(
            {
                "title": "Settings walkthrough",
                "summary": "Configure the worker.",
                "steps": [
                    {
                        "index": 1,
                        "heading": "Open settings",
                        "instruction": "Open the settings panel.",
                        "timestampStart": 0,
                        "timestampEnd": 3,
                        "screenshot": "step-001.jpg",
                        "evidence": "visual: settings panel",
                        "confidence": 0.82,
                    }
                ],
                "warnings": ["Review generated copy before publishing."],
            }
        )

        self.assertIn("![Open settings](assets/step-001.jpg)", markdown)
        self.assertIn("Confidence: 82%", markdown)
        self.assertIn("## Review Notes", markdown)


if __name__ == "__main__":
    unittest.main()
