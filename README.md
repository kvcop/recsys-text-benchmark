# Recsys Text Benchmark: TF-IDF vs Ollama Embeddings

Этот репозиторий показывает на воспроизводимом примере, как сравнить два подхода к
текстовым признакам в рекомендательной задаче:

1. `TF-IDF`
2. `Embeddings` через `Ollama` (CPU/GPU)

Цель проекта: не «доказать, что один метод всегда лучше», а дать практичную методику
и артефакты, чтобы принять инженерное решение по качеству, задержке и стоимости.

## Что внутри и зачем

Скрипт бенчмарка (`src/recsys_text_benchmark/benchmark.py`) делает честный A/B:

1. Фиксирует постановку задачи и ранжировщик.
2. Меняет только тип текстовых признаков.
3. Считает ranking-метрики и time-метрики.
4. Разделяет cold-start и warm-кейс через кэш эмбеддингов.

## Датасет и продуктовый контекст

Используется `MIND small` (news recommendation):

1. `news.tsv`: карточки новостей (title, abstract и др.).
2. `behaviors.tsv`: показы пользователю с кандидатами и метками клика (`0/1`).

Размеры `MIND small` на нашем прогоне:

1. `train`: 51,282 новостей, 156,965 impressions.
2. `dev`: 42,416 новостей, 73,152 impressions.

Что это означает для UX:

1. Пользователь видит ленту кандидатов.
2. У пользователя есть история кликов/чтений.
3. Система должна отсортировать кандидатов так, чтобы релевантные новости оказались выше.

По dev-части видно типичную «разреженность» рексиса:

1. В среднем ~37.47 кандидата на impression.
2. В среднем ~1.52 позитивных клика на impression.
3. Доля позитивов ~4.06%.

## Какая подвыборка использовалась

Для быстрых и воспроизводимых сравнений применялась подвыборка impressions:

1. `seed=42` для стабильности выборки.
2. `max_history=50` (ограничение длины пользовательской истории).
3. Фильтры:
   - история не пустая;
   - в impression есть и позитивы, и негативы.

Основные конфигурации:

1. `50 impressions`: быстрый cold-start CPU/GPU и sanity-check качества.
2. `3000 impressions`: расширенная проверка стабильности на большем срезе.

Почему так:

1. На полном dev наборе cold-start эмбеддингов слишком долгий для итеративной разработки.
2. Малый срез нужен для быстрой инженерной обратной связи.
3. Больший срез нужен, чтобы не опираться только на «игрушечный» результат.

## Методика оценки

Фиксируем всё, кроме фичей:

1. Одинаковые impressions.
2. Одинаковая формула скоринга.
3. Одинаковая агрегация пользовательского профиля.

Детали скоринга:

1. Профиль пользователя: средний вектор по истории кликов.
2. Score кандидата: cosine/dot similarity профиля и кандидата.
3. Отличается только источник векторов: `TF-IDF` или `embeddings`.

Метрики качества:

1. `MRR`
2. `NDCG@5`, `NDCG@10`
3. `Recall@5`, `Recall@10`

Метрики времени:

1. `feature_time_sec`: время подготовки признаков за весь прогон.
2. `eval_time_sec`: время ранжирования за все impressions.
3. `total_time_sec`: сумма.

Важно: это время **за весь прогон**, не за один API-запрос.

## Ключевые результаты (на 50 impressions)

Качество (`TF-IDF` vs `Embeddings`):

1. `MRR`: `0.2836 -> 0.3197` (`+12.74%`)
2. `NDCG@10`: `0.3058 -> 0.3681` (`+20.40%`)
3. `Recall@10`: `0.4987 -> 0.6313` (`+26.60%`)

Скорость cold-start (`Embeddings CPU` vs `Embeddings GPU`):

1. `feature_time_sec`: `741.231s -> 21.844s` (`~33.93x` быстрее на GPU)
2. `eval_time_sec`: около `0.003s` в обоих режимах (узкое место не ранжирование, а фичи)

Практический вывод:

1. Эмбеддинги дают лучшее ранжирование.
2. Для production их выгодно использовать с offline precompute и кэшированием item-векторов.
3. TF-IDF стоит держать как baseline/fallback и быстрый старт.

## Рекомендуемая архитектура для сервиса

1. Offline: батч-пайплайн пересчёта item-эмбеддингов.
2. Storage: feature store или векторный индекс.
3. Online: считаем только user/query-вектор и делаем скоринг по готовым item-векторам.
4. Контроль: A/B с `CTR`, `NDCG@K`, `p95 latency`, стоимость инфраструктуры.

## Структура репозитория

```text
recsys-text-benchmark/
  src/recsys_text_benchmark/
    benchmark.py          # основной CLI-скрипт эксперимента
    __init__.py
  README.md
  pyproject.toml
  .gitignore
```

Генерируемые артефакты во время запусков:

1. `data/` или `data_hostgpu/` — скачанный датасет и кэш эмбеддингов.
2. `results/` — json-отчёты прогонов.

Они намеренно не коммитятся в Git по умолчанию.

## Быстрый запуск

```bash
cd ~/code/mine/recsys-text-benchmark
uv sync
```

### Вариант A: локальный Ollama

```bash
ollama serve
ollama pull nomic-embed-text
```

### Вариант B: Ollama в Docker с GPU

```bash
docker run -d --gpus all -p 11435:11434 \
  --name ollama-gpu-bench \
  -v /home/user/.ollama:/root/.ollama \
  ollama/ollama:latest
```

## Запуски экспериментов

Только TF-IDF:

```bash
uv run recsys-bench --mode tfidf --max-impressions 3000 --seed 42
```

Embeddings через локальный Ollama:

```bash
uv run recsys-bench --mode embeddings --max-impressions 50 --seed 42 \
  --ollama-model nomic-embed-text:latest --ollama-base-url http://127.0.0.1:11434
```

Embeddings через Docker Ollama:

```bash
uv run recsys-bench --mode embeddings --max-impressions 50 --seed 42 \
  --ollama-model nomic-embed-text:latest --ollama-base-url http://127.0.0.1:11435
```

Оба режима сразу:

```bash
uv run recsys-bench --mode both --max-impressions 1000 --seed 42
```

## Что смотреть в выводе

1. `results/summary*.json`:
   - `dataset` (что именно было оценено),
   - `results.tfidf` / `results.embeddings`,
   - метрики качества и времени.
2. `loaded_from_cache`:
   - `false` означает cold-start;
   - `true` означает warm/cached сценарий.

## Скрипты и код

Основной скрипт — `benchmark.py`. В коде:

1. Русские docstring у функций и классов.
2. Комментарии в местах, где логика может быть неочевидной:
   - фильтрация impressions,
   - расчёт ranking-метрик,
   - кэширование эмбеддингов,
   - разделение feature/eval времени.
