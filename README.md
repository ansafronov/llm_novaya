# Novaya archive → Qwen3.5-4B

## Пилот по авторскому языку

Модуль `novaya.editorial_style` выбирает характерный фрагмент из соседних
абзацев или пропускает статью. На этапе отбора текст передаётся целиком
или последовательными блоками без перекрытия (`--chunk-chars`, по умолчанию
24 000 символов текста). Для длинной статьи берётся первый подходящий
фрагмент. Markdown-разметка, заголовки и адреса
ссылок убираются до отбора. Авторская формулировка и видимый текст
встроенных ссылок сохраняются. Локальная модель составляет сухой список
сведений, а обучающая цель берётся из очищенного авторского фрагмента.

```bash
.venv/bin/python -m novaya.editorial_style --port 8083 \
  --sample runs/editorial/qwen38-lead-selection-heldout/sample.jsonl \
  --output runs/editorial/qwen38-style-facts
```

В `candidates.jsonl` сохраняется решение по каждой статье. Перед использованием
пары сверяйте выразительность цели, очистку и достаточность карточки по источнику.

## Пилот: обучение по карточкам фактов

Нужен CUDA-сборка `llama-server` и файл `Qwen3-14B-Q4_K_M.gguf`
из репозитория Qwen. В отдельном терминале:

```bash
llama-server -m /path/to/Qwen3-14B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 -np 1
```

Из корня проекта:

```bash
.venv/bin/python -m novaya.editorial --output runs/editorial/pilot
```

В `review.jsonl` — до 30 пар с карточками, исходными предложениями и двумя
целями; в `audit.jsonl` — удалённые блоки и время по статьям. Проверьте
исходный текст и атрибуцию, достаточность фактов, копирование и ложные
удаления. Отдельно проверьте VRAM через `nvidia-smi` во время разметки.
Требуется минимум 24 пригодных пары из 30 для выбранной цели и не более
одного ошибочно удалённого содержательного блока.

Задание к карточке сформулировано как просьба написать начало или
фрагмент заметки для «Новой газеты».
Для авторских фрагментов целевой ответ взят из архива. Для синтетических
лидов ту же стилевую команду получает модель-разметчик. Обычные диалоги
готовятся без этой команды.

После проверки тем же модулем подготовьте отдельные каталоги для
`--split train` и `--split validation`, увеличив `--articles` по выходу
пилота. Для двух вариантов используйте одинаковые исходные фрагменты.
Диалоговый файл должен содержать проверенные JSONL-записи `{"messages":[...]}`
с последним сообщением `assistant`; укажите источник, версию и лицензию.

```bash
.venv/bin/python -m novaya.editorial_training fragment \
  --pilot runs/editorial/pilot --train runs/editorial/train \
  --validation runs/editorial/validation --dialogues data/dialogues.jsonl \
  --dialogue-source 'источник и версия' --dialogue-license 'лицензия' \
  --approved-pairs 24 --incorrect-removals 0 --output runs/editorial/fragment
```

Подставьте реальные результаты проверки вместо чисел из примера. Для
синтетических лидов используйте `lead` и отдельный каталог. Перед обучением
остановите сервер разметки. Выполните команды по порядку:

```bash
cfg=runs/editorial/fragment/config.json
.venv/bin/python -m novaya.cli --config "$cfg" evaluate
.venv/bin/python -m novaya.cli --config "$cfg" train --smoke
.venv/bin/python -m novaya.cli --config "$cfg" train
```

Сравнивайте ответы базы, старого адаптера и новых чекпоинтов вслепую:
15 лидов, 15 авторских фрагментов и 30 обычных заданий, включая уточнения.
Чтобы получить обезличенные ответы, запустите:

```bash
.venv/bin/python -m novaya.editorial_compare --validation runs/editorial/validation \
  --dialogues data/dialogues-validation.jsonl --output runs/editorial/comparison \
  --adapter runs/qwen35/adapter --adapter runs/editorial/fragment/run/adapter
```

Оценивайте `review.jsonl` до открытия `key.json`. Для сравнения промежуточных
чекпоинтов добавьте их пути отдельными аргументами `--adapter`. Если обучаете
оба варианта, передайте `--paired` при сборке обоих конфигов.
Чтобы отдельно измерить качество обычных диалогов, замените
`--validation runs/editorial/validation` на `--dialogues-only`.

Отдельно отмечайте неподтверждённые утверждения, повторы и копирование.
Принимайте адаптер, если в своём редакционном режиме он выигрывает чаще,
чем проигрывает базе, не чаще добавляет неподтверждённые факты и решает
не более чем на два обычных задания меньше.

QLoRA SFT по заголовку, жанру и рубрике. Используются только исходные статьи. Подзаголовок и начало статьи в запрос не попадают. Адаптер хранится отдельно и отключается.

## Запуск

Нужны рабочий NVIDIA driver, CUDA и GPU с BF16 (цель — RTX 4060 Ti 16 GB). GPU доступен при запуске вне песочницы: RTX 4060 Ti, driver 580.95.05. В песочнице CUDA скрыта; её отрицательная проверка не означает неисправность драйвера. Обучение и качество подтверждаются только артефактами успешных запусков.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/novaya prepare --fetch
.venv/bin/novaya train --smoke
.venv/bin/novaya evaluate --output-name baseline
.venv/bin/novaya generate --suite
.venv/bin/novaya train
.venv/bin/novaya evaluate --adapter runs/qwen35/adapter --output-name adapted
.venv/bin/novaya generate --adapter runs/qwen35/adapter --suite
```

Если архив уже скачан: `novaya prepare --archive data/archive`. Partial clone + sparse checkout получает только `articles/**/*.md`, без изображений. Исходный commit записывается в `data/prepared/report.json`. Не меняйте источник после подготовки. Управляющие символы C1, которыми в архиве записана пунктуация CP1252, восстанавливаются до разбора шапки и удаления повторного заголовка. Минимум содержательного текста — 50 слов; это явный параметр очистки в `novaya/data.py`.

Начальный контекст 4096 дал OOM; текущая конфигурация использует предусмотренный планом fallback 2048. Диагностика: `runs/qwen35/oom-4096.json`.

Конфигурация: `configs/default.json`; альтернативная — `novaya --config path.json ...`. При первом model load фиксируется фактический SHA в `runs/qwen35/provenance.json`. Точные установленные версии зафиксированы в корневом `requirements.lock.txt`; импорты библиотек и CUDA проверены. Установка этого окружения: `.venv/bin/pip install -r requirements.lock.txt`. Точные установленные версии сохраняются в `requirements.lock.txt` при загрузке модели; GPU-совместимость подтверждает только успешный smoke.

Для всей последовательности с автоматическим fallback 4096 → 2048: `.venv/bin/python scripts/run_experiment.py`. Для быстрого поддерживаемого линейного attention установите `.venv/bin/pip install -e ".[kernels]"` (FLA/Triton); полный SDPA не навязывается.

Smoke выполняет 20 optimizer steps (320 длинных примеров), backward, validation, сохранение и повторную загрузку адаптера. При OOM процесс прекращается: запустите `novaya train --smoke --context 2048` в новом процессе, затем установите `context: 2048` в конфигурации и повторите baseline и генерации. Повторный OOM означает остановку с диагностикой, без замены модели. Полное обучение — одна эпоха. Продолжение: `novaya train --resume` или `--resume runs/qwen35/checkpoints/checkpoint-N`.

## W&B

Включён по умолчанию, проект `novaya-qwen35`. Положите ключ в `.wandb/token` (также поддерживается `~/.wandb/token`) или задайте `WANDB_API_KEY`. Ключ читается только при запуске и не записывается в конфигурацию, метрики или артефакты; `.wandb/` исключена из Git. Удобно ограничить права файла: `chmod 600 .wandb/token`.

W&B получает параметры эксперимента, loss, learning rate и промежуточные validation NLL/PPL, итоговые метрики smoke/обучения, скорость и пик VRAM. `evaluate` создаёт отдельный run с полной validation. Тексты статей и сравнения генераций остаются локально. `WANDB_MODE=offline` сохраняет локальные логи; `wandb.enabled: false` полностью отключает интеграцию. Не включайте автоматическую загрузку кода или моделей с секретами.

## Оценка и артефакты

Train строго раньше validation, даты не делятся, точные нормализованные дубликаты оставляют раннюю публикацию. Разбиение зафиксировано JSONL до токенизации. Промежуточная выборка — до 512 равномерно расположенных по времени статей; финальная — вся validation. Target — текст статьи и `<|im_end|>`; prefix и padding исключены из loss. Обрезка справа не добавляет EOS. Доля обрезаний записывается отдельно.

PPL вычисляется как exp(sum NLL / sum target tokens), logits между примерами не сохраняются. Лучший checkpoint выбирается по такому validation NLL. Проверки раз в 500 optimizer steps и в конце эпохи. BF16 NF4 double quantization, LoRA r16/alpha32/dropout0.05 только на языковых линейных слоях, без lm_head; batch1/accum16, AdamW8bit, lr1e-4, warmup3%, cosine, seed42, packing отсутствует. Attention выбирает штатная реализация Transformers для Qwen3.5, без принудительного SDPA.

`generations-base.jsonl` и `generations-adapter.jsonl` содержат 20 одинаковых заголовков validation и 20 задач из `configs/general_tasks.json`, фиксированный seed, показатели повторов и совпадений восьмисловных фрагментов с исходником. Оцените попарно композицию, абзацы, лексику, связность, повторы, заимствования; для общих заданий — правильность и соблюдение инструкций. Автоматические совпадения — вспомогательный показатель, не оценка достоверности. Снижение PPL и сохранение универсальности заранее не гарантируются.

Артефакты: `data/prepared/` (статьи, split, отчёт), `runs/qwen35/` (provenance, tokenizer/chat template, lock, LoRA targets, checkpoints, adapter, метрики и генерации). Для нового эксперимента используйте новый output. Базовая модель могла видеть архив до нашего SFT; генерация событий и цитат по заголовку не обеспечивает их достоверности.

Референсы: [архив](https://github.com/novaya-gazeta-ru/archive), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

Сценарий `scripts/run_experiment.py` повторно использует baseline и 40 генераций при совпадении identity, возобновляет обучение с последнего полного checkpoint. Помимо основных сохранений/evaluation каждые 500 шагов, `recovery_save_steps: 50` сохраняет состояние для восстановления без дополнительного evaluation.

Опциональный подбор микробатча: `novaya probe-batch` проверяет 2 → 1 на длинных исходных статьях, включая backward, AdamW8bit step и evaluation; модель/optimizer/DataLoader создаются заново внутри каждой попытки Accelerate. `novaya train --smoke --auto-batch` проверяет выбранный размер на 20 шагах; затем `novaya train --auto-batch` применяет его с неизменным эффективным batch. Это не автоматический перезапуск всей эпохи. Результаты: `batch-plan.json`. Стандартный микробатч остаётся 1.

Каждый W&B-run сохраняет `launch-parameters` artifact: команду, эффективную конфигурацию, фиксированные параметры рецепта, версии зависимостей, revisions и manifest данных. Для обучения дополнительно сохраняется artifact `training-parameters` с фактическими TrainingArguments. Копии: `runs/qwen35/launches/<run-id>/`; ключи доступа редактируются, переменные окружения целиком не выгружаются.
