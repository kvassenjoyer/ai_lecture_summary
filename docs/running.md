# Установка и эксплуатация Лекториума

Команды выполняются в PowerShell из корня проекта. Первая установка и обычный запуск описаны в [README](../README.md). Здесь собраны дополнительные настройки и варианты эксплуатации.

## Новая установка на Windows

Нужны Python 3.12, менеджер окружения `uv`, FFmpeg вместе с `ffprobe` и Node.js для YouTube. Python может установить `uv`; Docker Desktop требуется для PostgreSQL/RustFS, но не для SQLite и локальных файлов.

```powershell
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
winget install --id OpenJS.NodeJS.LTS -e
```

После установки откройте новый PowerShell в папке проекта:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:UV_PYTHON = '3.12'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
.venv\Scripts\python.exe scripts\doctor.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -OpenBrowser
```

`bootstrap.ps1` создаёт `.venv`, устанавливает зависимости и скачивает Whisper/Qwen с проверкой контрольных сумм. Windows-окружение использует CUDA-сборку PyTorch; доступность GPU определяется во время работы. Установка требует сети и места для весов, временных частей и зависимостей.

Повторный запуск загрузчика обращается к Hugging Face и может выбрать новую ревизию модели. Это не автономная проверка существующего кэша. Полученные ревизии и хеши записываются в `models/manifest.json`; для воспроизведения проверки сохраняйте этот файл вместе с соответствующим кэшем.

## Профили запуска

| Профиль | Где работают API и модели | Данные |
|---|---|---|
| Подготовленная копия | Windows, CUDA при доступности | PostgreSQL и RustFS в Docker |
| Новая `.env.example` | Windows, автоматический выбор устройства | SQLite и `data/uploads`, Docker не нужен |
| Полностью контейнерный | API и worker в Docker, CPU | PostgreSQL и RustFS в Docker |

Профиль определяется `.env`, а не наличием Docker на компьютере. Настройки API и worker должны указывать на одну БД и одно хранилище. При `local` им нужен общий каталог данных.

Для PostgreSQL/S3 установите и запустите Docker Desktop, используйте блок настроек в конце `.env.example` и запустите:

```powershell
docker compose up -d --wait postgres objectstore
```

Переключение `DATABASE_URL` и `STORAGE_BACKEND` не переносит существующую историю и аудио. В подготовленной `.env` параметр `COMPOSE_PROJECT_NAME` сохраняет связь с существующим Docker-окружением после переименования проекта.

Полностью контейнерный вариант предполагает заранее скачанный кэш `models`:

```powershell
docker compose up -d --build postgres objectstore api worker
```

Перед этим остановите Windows API и worker, чтобы освободить порт и исключить конкуренцию обработчиков. GPU в Compose не подключён. Сборка образа проверена, полный inference на CPU - нет; этот вариант не заменяет подтверждённый Windows GPU-профиль.

## Настройки

Основной файл - `.env`, образец - [.env.example](../.env.example). Переменные окружения имеют приоритет; после изменения настроек перезапустите API и worker.

| Параметры | Назначение |
|---|---|
| `DATABASE_URL`, `STORAGE_BACKEND`, `DATA_DIR` | БД, тип хранилища и рабочие каталоги |
| `S3_ENDPOINT_URL`, `S3_BUCKET`, ключи `S3_*` | Подключение к S3-совместимому хранилищу |
| `MODEL_DEVICE=auto`, `QWEN_LOAD_IN_4BIT=true` | Выбор CPU/CUDA; 4-битная загрузка Qwen используется на CUDA |
| `HF_HOME=models`, `MODEL_LOCAL_FILES_ONLY=true` | Каталог моделей и работа с уже скачанными весами |
| `MAX_UPLOAD_MB=500`, `MAX_AUDIO_SECONDS=14400` | Предельный размер в МиБ и длительность в секундах |

Все параметры пайплайна и очереди перечислены в [архитектуре](architecture.md). Если меняете `HF_HOME`, загрузите модели в тот же каталог через `download_models.py --cache-root`; сам bootstrap использует каталог `models`.

## Остановка, данные и журналы

`start.ps1` запускает один API и один worker в фоне, записывает их идентификаторы в `data/processes.json` и ждёт ответа API. Автоматической перезагрузки исходного кода нет. Изменения Python требуют перезапуска; изменения статических файлов - обновления страницы.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop.ps1
docker compose down
```

Скрипт проверяет принадлежность процессов перед остановкой. `docker compose down` сохраняет тома; вариант с `--volumes` удаляет данные контейнеров. Для резервной копии нужны и БД с результатами, и аудиохранилище; один кэш моделей историю не содержит.

| Где смотреть | Содержание |
|---|---|
| `logs/api.log`, `logs/api-error.log` | Запросы и ошибки API |
| `logs/worker.log`, `logs/worker-error.log` | Обработка, модели и подробности ошибок задач |
| `data/` | Служебные файлы; в локальном профиле также БД и аудио |
| `models/hub`, `models/manifest.json` | Кэш моделей и сведения о скачивании |

## Диагностика и тесты

```powershell
.venv\Scripts\python.exe scripts\doctor.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
```

`doctor.py` показывает окружение, CUDA, найденные конфигурации моделей и свободное место. Он не проверяет все веса, подключения к PostgreSQL/S3 или работоспособность worker. `/health` проверяет API и БД, но не готовность моделей.

Обычные тесты подменяют тяжёлые модели и внешние сервисы; два инфраструктурных теста по умолчанию пропущены. Для их отдельного запуска нужны PostgreSQL/RustFS со стандартными локальными настройками, остановленный обычный worker и пустая очередь:

```powershell
docker compose up -d --wait postgres objectstore
$env:RUN_INFRA_TESTS = '1'
try { .venv\Scripts\python.exe -m pytest tests/test_infrastructure.py -q }
finally { Remove-Item Env:RUN_INFRA_TESTS }
```

| Симптом | Что проверить |
|---|---|
| Порт 8000 занят | Откройте уже работающий сервис; для перезапуска используйте `stop.ps1` |
| API доступен, задача остаётся в очереди | Запущен ли worker и совпадают ли его БД/хранилище с API |
| Модели не найдены | Путь `HF_HOME`, наличие кэша и манифеста; вывод `doctor.py` |
| Ошибка YouTube | Публичность видео, доступность сети, Node.js в PATH и журнал worker |
| Задача завершилась ошибкой | Сообщение в интерфейсе и `logs/worker-error.log`; при необходимости отправьте источник заново |
