# SKILLS.md

This repo ships four Claude Code skills under `.claude/skills/` that wrap the
service's common workflows. Each skill is a procedural guide the agent follows
when you ask for that kind of work — they encode the conventions in CLAUDE.md
and the deeper docs in `readmes/`.

| Skill | When to use it |
|---|---|
| [`new-app`](.claude/skills/new-app/SKILL.md) | Scaffold a new MDX analytics app under `apps/<name>/` following the `BaseApp` pattern. Trigger: "create a new app", "add a new analytics app", "scaffold an app". |
| [`new-incident`](.claude/skills/new-incident/SKILL.md) | Add a new frame-level incident type to `FrameStateMgmt` in `frame_state_management.py`. Trigger: "add a new incident type", "add a new violation detection". |
| [`implement-feature`](.claude/skills/implement-feature/SKILL.md) | Drive an open-ended feature end-to-end (understand → plan → code → tests → docs → pre-flight). Trigger: any feature request not covered by a more specific scaffold skill. |
| [`run-test`](.claude/skills/run-test/SKILL.md) | Run the full test pipeline — unit tests first, then the Docker-Compose integration test — and report results. Trigger: "run tests", "verify the build", "test the pipeline end-to-end". |

See [CLAUDE.md](CLAUDE.md) for the full agent guide and [readmes/](readmes/) for
human-facing documentation.
