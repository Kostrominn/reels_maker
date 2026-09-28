# Multicam auto-switch (agent notes)

This change adds an audio-energy-based multicam planner + gated audio mix for setups where each speaker has their own mic and camera. It does **not** modify sync code; it only **reads** `offsets.json`.

## What was added
- New module: `src/multicam.py`
  - Builds a shotlist by comparing per-camera audio energy on the **ref** timeline.
  - Builds a gated audio mix (active mic = 0 dB, inactive mics = ducked).
  - Optionally renders the final video by cutting segments and concatenating.
- CLI command: `multicam` added to `main.py`.

## Key behavior
- **No wide shot**: when nobody is clearly speaking, it holds the last active camera.
- **Min shot length**: uses `video.min_shot_duration` from `config.yaml`.
- **Offsets sign convention**: same as `sync.py`.
  - If `offset > 0`, `other` is delayed.
  - For mapping ref time to other camera: `t_other = t_ref + offset`.
  - Drift is respected if `fit_drift` exists: `offset(t) = intercept + slope * t`.

## How to run
You need offsets for every "other" camera vs the same ref (e.g., 14 as ref, 15/16 as others).
If `offsets.json` contains `ref`/`other`, you can omit `--ref/--other`.

Example (shotlist + mix + render):
```bash
python main.py multicam \
  --offsets storage/sync/offsets_14_15.json \
  --offsets storage/sync/offsets_14_16.json \
  --start-seconds 0 \
  --duration-seconds 60
```

Plan-only (no render):
```bash
python main.py multicam \
  --offsets storage/sync/offsets_14_15.json \
  --offsets storage/sync/offsets_14_16.json \
  --start-seconds 0 \
  --duration-seconds 60 \
  --plan-only
```

Outputs (written to `storage/output/`):
- `*_shotlist_sX_eY.json`
- `*_mix_sX_eY.wav`
- `*_multicam_sX_eY.mp4` (if not `--plan-only`)

## Tuning knobs
- `--speech-margin-db` (default 8): threshold above noise floor to consider speech.
- `--switch-margin-db` (default 4): how much louder the new speaker must be.
- `--hop-seconds` (default 0.1): analysis step.
- `--frame-ms` (default 200): analysis window.
- `--duck-db` (default 18): how much to attenuate inactive mics in the mix.

## Speaker-ID mode (2026-02-06)
Добавлен режим `--dominance-mode speaker` — выбор камеры по speaker-ID вместо энергии.

Что делает:
- Строит speaker timeline по коротким окнам (по умолчанию `--speaker-window-seconds 1.5`).
- Использует cosine similarity к референсам.
- Учитывает `--min-shot-seconds` и `--hop-seconds` (рекомендуется 0.5s).
- Энергетический гейт остаётся: если речи нет, держит текущую камеру.

Референсы:
- `--speaker-refs path.json` (JSON/YAML с samples: label, video_path, start_seconds, end_seconds)
- или `--speaker-ref-dir storage/audio/speaker_refs` (WAVs с префиксом label_*.wav)

Пример:
```bash
python main.py multicam \
  --offsets storage/sync/offsets_14_15.json \
  --ref storage/audio/IMG_0014.wav \
  --other storage/audio/IMG_0041.wav \
  --dominance-mode speaker \
  --speaker-ref-dir storage/audio/speaker_refs \
  --speaker-map andrey=IMG_0014 \
  --speaker-map me=IMG_0041 \
  --speaker-window-seconds 1.5 \
  --speaker-lead-seconds 0.25 \
  --hop-seconds 0.5
```

Доп. выход:
- `*_speaker_timeline_sX_eY.json`

## Limitations / TODOs
- No automatic rescale/crop if cameras have different resolutions.
- Concatenation uses `ffmpeg -c copy`; mismatched codecs/params across cameras can break concat.
- No smoothing across segment boundaries (could add short crossfades).
- If ref duration cannot be probed, user must pass `--end-seconds`.

## Files touched
- `src/multicam.py` (new)
- `main.py` (imports + CLI + command handler)

---

## Sync alignment notes (2026-02-05)

Goal: make camera audio sync automatic, non-destructive, and repeatable for other video pairs.

### What was implemented/changed
- Added long-range sync features and caching in `src/sync.py`:
  - Chunked audio cache (`sync-cache`) to avoid repeated network extraction.
  - Drift fitting across multiple windows (`--fit-drift`) with outlier filtering.
  - Residual-based offset refinement (`--refine-offset`).
  - Audio-only verification/auto-correct (`sync-apply --audio-only --auto-correct`).
- Added CLI flags and commands in `main.py`:
  - `sync-cache`, `sync-apply`, plus `--fit-*`, `--refine-*`, `--cache-dir`.
- Cache index now persists across partial runs (does not overwrite when a run fails).

### Current state
- Partial cache exists in `storage/sync/cache/` for the two test videos.
- Last computed offset is stable around **-19.82 s** (drift ~0.076 s/hour over the cached range).
- Latest results are stored in `storage/sync/offsets.json`.
- Verification outputs live in `storage/sync/aligned/` (no source files are modified).

### How to run (repeatable)
1. Cache audio chunks (non-destructive):
```bash
python main.py sync-cache REF.mov OTHER.mov \
  --chunk-seconds 600 --fast-seek-only --cache-dir storage/sync/cache
```

2. Estimate offset + optional drift fit:
```bash
python main.py sync-offset REF.mov OTHER.mov \
  --window-seconds 120 --max-lag-seconds 30 --start-seconds 780 \
  --fit-drift --fit-start 0 --fit-duration 1200 --fit-step-seconds 120 \
  --fit-min-score 3 --fit-outlier-seconds 0.6 \
  --cache-dir storage/sync/cache
```

3. Verify/auto-correct and generate check WAVs:
```bash
python main.py sync-apply REF.mov OTHER.mov \
  --offset-seconds -19.82 \
  --audio-only --audio-window-seconds 60 \
  --verify-start-seconds 812 --verify-window-seconds 30 \
  --auto-correct --auto-correct-iters 2 \
  --cache-dir storage/sync/cache
```

### Next actions (pipeline)
- Extend cache to cover the full video duration for both files.
- Re-run `sync-offset --fit-drift` on the full range to confirm drift.
- Keep a single trim at the start if drift stays small (< ~0.2 s/hour).
- Generate offsets for all other cameras vs the same reference.
- Use `multicam` to build a shotlist + gated audio mix, then render reels.
