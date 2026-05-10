"""Prompt templates for OpenLoom inference services."""

QWENVL_SYSTEM = (
    "You are a screen recording analyst that produces structured software tutorials. "
    "Given a screen recording, identify each distinct UI action or state change. "
    "For each step, provide a concise description of what happened on screen."
)

QWENVL_USER = (
    "Analyze this screen recording as a software tutorial. "
    "Return concise timestamped steps describing visible UI actions and state changes."
)

WHISPER_USER = (
    "Transcribe this recording. Return only the spoken words, with no commentary."
)
