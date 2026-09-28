# Автоматический пайплайн `main.py`

Первая версия инструмента: транскрибация (faster-whisper или GigaAM), поиск интересных моментов через LLM, нарезка, переключение камер по громкости, публикация через Instagram Graph API и статистика по выложенным рилсам. Этим пайплайном нарезаны первые рилсы (весна 2026). Для роликов с точными субтитрами и проверками дальше использовался модуль [`interview_reels`](../interview_reels/README.md).

Исходный план разработки и заметки — в [`docs/notes/`](notes/).

## Требования

- Python 3.10+
- FFmpeg (должен быть установлен в системе)
- API ключ OpenAI-compatible провайдера (например, OpenRouter) или локальный Ollama
- Минимум 20GB свободного места

## Установка

### 1. Установка FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Установка Python зависимостей

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### 2.1. (Опционально) Качественная транскрибация через GigaAM-v3

GigaAM-v3 (особенно `v3_e2e_rnnt`) обычно даёт более “чистый” русский и пунктуацию. Источники:
- Статья: `https://developers.sber.ru/kak-v-sbere/culture/gigaAM-v3?ysclid=ml58c43kzk915119393`
- Репозиторий: `https://github.com/salute-developers/GigaAM`

Установка (в текущий venv):

```bash
pip install -r requirements.gigaam.txt
```

Важно: при первом запуске веса могут скачиваться с CDN. Если скачивание падает по таймауту — можно скачать вручную в кеш:

```bash
mkdir -p ~/.cache/gigaam
curl -L -o ~/.cache/gigaam/v3_e2e_rnnt.ckpt "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt.ckpt"
curl -L -o ~/.cache/gigaam/v3_e2e_rnnt_tokenizer.model "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt_tokenizer.model"
```

### 3. Настройка

Создайте `.env` файл:
```bash
cp .env.example .env
```

Выберите один из вариантов:

OpenRouter (рекомендуется для качества):
```bash
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-ваш-ключ
# Опционально:
# OPENROUTER_APP_NAME=reels-maker
# OPENROUTER_SITE_URL=https://example.com
```
Готовый конфиг: `config.openrouter.yaml`.
Для теста Gemma: `config.gemma.openrouter.yaml`.

Локально через Ollama (без внешнего API-ключа):
```bash
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434/v1
```
И запустите сервис:
```bash
ollama serve
```

Gemini (для сравнения; это не open-source):
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=ваш_ключ
```
Готовый конфиг: `config.gemini.yaml`.

## Использование

### Быстрый старт

```bash
# Полный пайплайн: от видео до рилсов
python main.py full-pipeline input_video.mov

# Или пошагово:
python main.py transcribe input_video.mov
python main.py analyze transcripts/input_video.json
python main.py generate clips/input_video_clips.json
```

### Транскрибация через GigaAM-v3 (внутри проекта)

Готовый конфиг: `config.gigaam.yaml`.

```bash
python main.py --config config.gigaam.yaml transcribe IMG_0043.mov --duration-seconds 20
```

Примечание: в самой библиотеке GigaAM метод `transcribe()` ограничен аудио ~до 25 секунд; для длинных файлов у них есть `transcribe_longform()` (требует доп. зависимостей и `HF_TOKEN`). В нашем проекте, если longform недоступен, будет fallback на “чанки” по `gigaam_chunk_seconds`.

### Локальный open-source LLM через Ollama

Готовый конфиг: `config.ollama.yaml`.

```bash
# Пример установки модели
ollama pull gemma2:2b

# Анализ транскрипта через локальную модель
python main.py --config config.ollama.yaml analyze storage/transcripts/IMG_0043_s0_d180.json
```

### Публикация финальных Reels в Instagram (Graph API)

Команда `instagram-upload` работает по готовому upload-паку (без рендера/перекодирования) и ведёт прогресс в CSV.

Подготовьте в `.env`:
```
INSTAGRAM_ACCESS_TOKEN=...
INSTAGRAM_IG_USER_ID=...
```

Проверка качества (без загрузки):
```bash
python main.py instagram-upload --qc-only --batch-size 0
```

Сухой прогон только для диапазона:
```bash
python main.py instagram-upload --dry-run --index-from 1 --index-to 10
```

Реальная публикация батчем по 5:
```bash
python main.py instagram-upload --index-from 1 --index-to 5 --batch-size 5
```

Примечания:
- По умолчанию уже опубликованные индексы из `UPLOAD_PROGRESS.csv` пропускаются.
- `--force-retry-failed` повторно пробует записи со статусом `failed`.
- `--strict-qc` блокирует загрузку, если у файла есть QC-предупреждения.

### Статистика по выложенным Reels (из MONTH_TRACKER CSV)

Команда `reels-stats` считает сводку по уже опубликованным рилсам (views/engagement) и показывает, какие строки ещё без метрик.

Быстрый запуск:
```bash
python main.py reels-stats --tracker-csv MONTH_TRACKER_20260311.csv
```

Если нужно отметить, что рилсы до конкретной даты уже выложены:
```bash
python main.py reels-stats --tracker-csv MONTH_TRACKER_20260311.csv --mark-posted-until 2026-03-23
```

Сухой прогон без записи в CSV:
```bash
python main.py reels-stats --tracker-csv MONTH_TRACKER_20260311.csv --mark-posted-until 2026-03-23 --no-write
```

Экспорт в Excel-friendly CSV (UTF-8 BOM + `;`):
```bash
python main.py reels-stats --tracker-csv MONTH_TRACKER_20260311.csv --excel-csv MONTH_TRACKER_20260311_excel.csv
```

Если Excel всё равно ломает кириллицу, используйте Unicode TSV (UTF-16 + табы):
```bash
python main.py reels-stats --tracker-csv MONTH_TRACKER_20260311.csv --excel-tsv MONTH_TRACKER_20260311_excel.tsv
```

### Структура данных

```
storage/
├── raw/           # Исходные видео (временно)
├── audio/         # Извлечённые аудио (временно)
├── transcripts/   # JSON с транскрипциями
├── clips/         # Метаданные о найденных фрагментах
└── output/        # Готовые рилсы
```

## Архитектура

Проект состоит из независимых модулей:

1. **Audio Extractor** — извлечение аудио из видео
2. **Transcriber** — транскрибация речи (faster-whisper или GigaAM)
3. **Speaker Detector** — определение говорящих (pyannote.audio)
4. **LLM Analyzer** — поиск интересных моментов (open-source или OpenAI-compatible API)
5. **Video Cutter** — нарезка видео по таймкодам
6. **Multi-cam Editor** — склейка с переключением камер
7. **Storage Manager** — управление дисковым пространством

## Разработка

### Структура проекта

```
reels_maker/
├── main.py              # CLI entry point
├── src/                 # Основные модули
│   ├── models.py
│   ├── audio_extractor.py
│   ├── transcriber.py
│   └── ...
├── prompts/             # Промпты для LLM
├── tests/               # Тесты
└── storage/             # Рабочие файлы
```

### Тестирование

```bash
pytest tests/
```
