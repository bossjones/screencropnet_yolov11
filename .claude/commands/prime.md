---
model: opus
description: Initial setup prompt - reads project structure and gains context
---

# Prime

## Purpose

Load project context by reading specs, agents, commands, skills, hooks, and tracked files to understand the current state of the `screencropnet_yolov11` (Twitter Screenshot Detection) codebase.

## Workflow

1. Read `specs/init.md`
2. Read `.claude/agents/*`
3. Read `.claude/commands/*`
4. Read `.claude/skills/*`
5. Read `.claude/hooks/*`
6. Read `justfile`
7. Run `git ls-files`
8. Report: "Project context loaded. Ready to assist with screencropnet_yolov11."
