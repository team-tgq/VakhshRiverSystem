# Project Run Guide

This project is configured to run with the conda environment:

`E:\anaconda\envs\VakhshRiverSystem`

## Quick Start

In Codex app, run the current branch from the current worktree root with:

```powershell
.\run_app.bat
```

This launches:

`C:\Users\TGQ\.codex\worktrees\0c9b\VakhshRiverSystem\main.py`

So what you see is the effect of the current branch/worktree changes, not the original project folder on `E:`.

Double-click:

`run_app.bat`

Or run in PowerShell:

```powershell
.\run_app.ps1
```

## PyCharm

Use the shared run configuration:

`Run VakhshRiverSystem`

It runs:

- script: `main.py`
- working directory: project root
- environment variables:
  - `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu --disable-gpu-compositing`
  - `QT_OPENGL=software`

## Optional

To run in offscreen mode for checks:

```powershell
.\run_app.ps1 -QtPlatform offscreen
```
