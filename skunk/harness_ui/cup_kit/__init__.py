"""OfficeQA Cup — external-team practice kit.

Self-contained client + practice harness. Mirrors the live cup's wire
protocol so an agent that runs against the practice harness will work
against the real competition server with no code changes — only the
``CUP_BASE_URL`` and ``CUP_TEAM_TOKEN`` env vars differ.
"""

VERSION = "0.1.6"
