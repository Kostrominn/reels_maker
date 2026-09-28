# START HERE — Vova Reels Agent

Прочитай этот файл первым. Цель: параллельно с прогоном Андрея поднять стабильный пайплайн по Вове (без скачивания исходников локально).

## 1) Контекст и рамки
- Проект: `/Users/mac/Documents/reels_maker`
- Текущий долгий раннер Андрея уже идёт в фоне и его не трогать.
- Источник Вовы: `disk:/Интервью/Вова`
- Уже найден рабочий sync для первой пары (подтвержден глазами по lip-sync):
  - `REF=IMG_0003`, `OTHER=IMG_0018`
  - `offset_start=-23.52`
  - файл: `/Users/mac/Documents/reels_maker/storage/sync/offsets_vova.json`
- Доступные видео:
  - `IMG_0003.mov`
  - `IMG_0004.mov`
  - `IMG_0005.mov`
  - `IMG_0007.mov`
  - `IMG_0018.mov`
  - `IMG_0019.mov`
  - `IMG_0020.mov`
  - `IMG_0021.mov`
  - `IMG_0024.mov`
  - `IMG_0025.mov`
  - `IMG_3665.mov`

## 2) Неподвижные стандарты (НЕ менять)
- Источники только `yadisk:...`, без ручного скачивания в локальные `.mov`.
- Для долгих задач использовать `config.lowmem.yaml`.
- Аудио в финальном мультикаме только `--audio-encoder alac` (AAC ранее давал треск).
- Монтаж: `--dominance-mode speaker`.
- Визуальный стиль заголовка: `--title-style overlay_center_dark` (без бегущей строки).
- Субтитры: `--subtitle-preset readable`.
- Для пары `0003/0018`: мягкий рефрейм камеры `0018` (чтобы убрать сильный сдвиг вправо):
  - `--other-vf "crop=w=iw/1.35:h=ih/1.35:x=iw-iw/1.35:y=(ih-ih/1.35)/2,scale=1080:1920,eq=gamma_r=1.12:gamma_g=1.02:gamma_b=0.88:saturation=1.07:contrast=0.97:brightness=0.005"`

## 3) Что сделать первым делом
1. Проверить список видео:
```bash
cd /Users/mac/Documents/reels_maker
./.venv/bin/python main.py --config config.preview.yaml yadisk-files \
  --path-prefix "disk:/Интервью/Вова" --limit 500 --media-type video
```
2. Найти рабочую пару камер по `sync-offset`.
3. Сохранить смещение в:
- `/Users/mac/Documents/reels_maker/storage/sync/offsets_vova.json`

## 4) Поиск пары камер (порядок)
1. Базовая проверка:
```bash
./.venv/bin/python main.py --config config.preview.yaml sync-offset \
  yadisk:disk:/Интервью/Вова/IMG_0018.mov \
  yadisk:disk:/Интервью/Вова/IMG_0019.mov \
  --auto-search --search-start 0 --search-duration 1800 --step-seconds 120 \
  --window-seconds 30 --min-score 3.0 --refine-offset --fast-seek-only
```
2. Если score/липсинк плохие, перебор:
- `IMG_0018 + IMG_0025`
- `IMG_0019 + IMG_0025`
- `IMG_0024 + IMG_0025`

3. После выбора пары записать файл:
```json
{
  "ref": "yadisk:disk:/Интервью/Вова/<REF>.mov",
  "other": "yadisk:disk:/Интервью/Вова/<OTHER>.mov",
  "offset_seconds": <float>
}
```

## 5) Рабочий рендер-профиль (финальный)
- `--dominance-mode speaker`
- `--audio-from-sources`
- `--audio-force-camera <OTHER_STEM>`
- `--audio-encoder alac`
- `--reaction-every-seconds 16.0`
- `--reaction-duration-seconds 2.2`
- `--pause-accel-profile medium`
- `--title-style overlay_center_dark`
- `--subtitle-preset readable`
- Цвет/тон:
  - `--ref-vf "eq=gamma_r=1.01:gamma_g=1.00:gamma_b=1.04:saturation=1.00:contrast=0.99"`
  - для `IMG_0018` использовать `--other-vf` из секции 2 (с crop+scale+eq)

Минимальный пример для `0003/0018`:
```bash
./.venv/bin/python main.py --config config.lowmem.yaml multicam \
  --ref yadisk:disk:/Интервью/Вова/IMG_0003.mov \
  --other yadisk:disk:/Интервью/Вова/IMG_0018.mov \
  --offsets storage/sync/offsets_vova.json \
  --dominance-mode speaker \
  --audio-from-sources \
  --audio-force-camera IMG_0018 \
  --audio-encoder alac \
  --ref-vf "eq=gamma_r=1.01:gamma_g=1.00:gamma_b=1.04:saturation=1.00:contrast=0.99" \
  --other-vf "crop=w=iw/1.35:h=ih/1.35:x=iw-iw/1.35:y=(ih-ih/1.35)/2,scale=1080:1920,eq=gamma_r=1.12:gamma_g=1.02:gamma_b=0.88:saturation=1.07:contrast=0.97:brightness=0.005" \
  --reaction-every-seconds 16.0 \
  --reaction-duration-seconds 2.2 \
  --pause-accel-profile medium \
  --title-style overlay_center_dark \
  --subtitle-preset readable \
  --start-seconds <START> \
  --end-seconds <END> \
  --title-text "<TITLE>" \
  --output storage/output/final_reels_vova/<FILE>.mov
```

## 6) Как запускать (10-мин чанками)
1. Делаем транскрипт чанка:
```bash
./.venv/bin/python main.py --config config.lowmem.yaml transcribe \
  yadisk:disk:/Интервью/Вова/<REF>.mov \
  --start-seconds <START> --duration-seconds 600
```
2. Переводим в absolute timeline (`*_abs.json`).
3. Запускаем `analyze`.
4. На каждом кандидате запускаем `multicam` с профилем из секции 5.

Выход складывать в:
- `/Users/mac/Documents/reels_maker/storage/output/final_reels_vova`

## 7) Критерии готовности
- Есть `records.ndjson` и `summary.json` у раннера.
- На каждом обработанном чанке есть готовые `.mov`.
- Проверка глазами: чистый звук, липсинк корректный, камера в основном на говорящем.
