# Reels Maker 🎬

Автоматическая генерация коротких рилсов из длинных видео интервью с использованием AI.

## Возможности

- 🎤 Транскрибация русской речи из видео (Whisper)
- 🤖 Автоматический поиск интересных моментов через LLM (open-source модели)
- 🎥 Поддержка мультикамерной съёмки
- ✂️ Автоматическая нарезка видео
- 🔄 Умное переключение между камерами
- 💾 Оптимизация хранения для больших файлов

## Статус проекта

⚠️ **В разработке** — см. [PLAN.md](PLAN.md) для детального плана

**Текущий прогресс**: 4%  
**MVP готов**: Нет

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

### Быстрый старт (когда MVP будет готов)

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

## FAQ

**Q: Сколько места нужно на диске?**  
A: Для обработки 1 часа видео с 3 камер нужно ~30GB. Промежуточные файлы удаляются автоматически.

**Q: Как долго обрабатывается 1 час видео?**  
A: Зависит от железа. На CPU ~20-30 минут, на GPU ~5-10 минут.

**Q: Поддерживаются ли другие языки?**  
A: Да, Whisper поддерживает 99 языков. Измените настройки в `config.yaml`.

## Roadmap

- [x] Базовая инфраструктура
- [ ] MVP: транскрибация + анализ + нарезка
- [ ] Speaker diarization
- [ ] Мультикамерность
- [ ] Web UI
- [ ] Batch processing

См. [PLAN.md](PLAN.md) для детального плана.

## Лицензия

MIT

## Контакты

По вопросам и предложениям создавайте Issues.
