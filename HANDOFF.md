# Handoff (Stable Lip-Sync + Clean Audio)

## Current Status
- Фоновый треск устранен.
- Рассинхрон в ветке `title` устранен.
- Проверка пройдена на `checkB/checkC`: синхрон корректный.

## Final Root Cause
- Проблема была в экспериментальной ветке титра через отдельные входы и `concat/setpts/anullsrc`.
- Эта схема периодически ломала таймстемпы и давала рассинхрон видео/аудио.

## Final Fix
- Ветка титра упрощена и стабилизирована:
  - Видео: `tpad` (черный префикс) + `drawtext`.
  - Аудио: `adelay` на длительность титра.
- Сегментный кэш исправлен: ключ кэша теперь учитывает `video_filter`,
  чтобы не переиспользовать старые (неверные по цвету) сегменты.
- Удалены черновые/экспериментальные пути:
  - `--camera-video-shift`
  - `--audio-from-segments`
- Сохранен рабочий режим:
  - `--audio-from-sources`
  - `--audio-force-camera IMG_0041`
  - `--audio-encoder alac`
- Пайплайн отбора клипов:
  - батчи по 10 минут для стабильной транскрибации
  - LLM-кандидаты: обычно 3-6 (не фикс 3)

## Final Working Command
```bash
./.venv/bin/python main.py --config config.preview.yaml multicam \
  --offsets storage/sync/offsets.json \
  --dominance-mode energy \
  --hop-seconds 0.4 \
  --frame-ms 200 \
  --min-shot-seconds 2.0 \
  --audio-crossfade-seconds 0 \
  --hard-gate \
  --audio-from-sources \
  --audio-force-camera IMG_0041 \
  --audio-encoder alac \
  --subtitle-transcript storage/transcripts/IMG_0014_s0_d2700.json \
  --subtitle-max-chars 22 \
  --subtitle-max-lines 2 \
  --title-seconds 1.6 \
  --title-font /System/Library/Fonts/Supplemental/Arial.ttf \
  --title-fontsize 56 \
  --ref-vf "eq=gamma_r=1.01:gamma_g=1.00:gamma_b=1.04:saturation=1.00:contrast=0.99" \
  --other-vf "eq=gamma_r=1.12:gamma_g=1.02:gamma_b=0.88:saturation=1.07:contrast=0.97:brightness=0.005" \
  --cache-segments \
  --cache-padding-seconds 2.0 \
  --start-seconds 901.14 \
  --end-seconds 960.02 \
  --title-text "Как перестать бояться Москвы и выбрать вуз за 5 минут" \
  --output storage/output/IMG_0014_multicam_s901.14_e960.02_v47_main_title_tpad.mov
```

## Verified New Renders
- `storage/output/IMG_0014_multicam_s1273.10_e1285.52_v47_checkB_title_tpad.mov`
- `storage/output/IMG_0014_multicam_s1320.10_e1339.06_v47_checkC_title_tpad.mov`
