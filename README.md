# Recsys Text Benchmark: TF-IDF vs Ollama Embeddings

Этот репозиторий показывает на воспроизводимом примере, как сравнить два подхода к
текстовым признакам в рекомендательной задаче:

1. `TF-IDF`
2. `Embeddings` через `Ollama` (CPU/GPU)

Цель проекта: не «доказать, что один метод всегда лучше», а дать практичную методику
и артефакты, чтобы принять инженерное решение по качеству, задержке и стоимости.

## Контекст дискуссии

Этот репозиторий вырос из обсуждения идеи «взять тот же ранжировщик, но заменить
`TF-IDF` на embeddings». Оригинальный контекст на Habr:

- https://habr.com/ru/articles/990386/

## Что внутри и зачем

Скрипт бенчмарка (`src/recsys_text_benchmark/benchmark.py`) делает честный A/B:

1. Фиксирует постановку задачи и ранжировщик.
2. Меняет только тип текстовых признаков.
3. Считает ranking-метрики и time-метрики.
4. Разделяет cold-start и warm-кейс через кэш эмбеддингов.
5. Поддерживает два ранжировщика:
   - `similarity` (исторический baseline),
   - `svm` (`LinearSVC`, чтобы аутентично проверить гипотезу «тот же SVM на других фичах»).

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

Фиксируем всё, кроме фичей и выбранного ранжировщика:

1. Одинаковые impressions.
2. Одинаковый train/dev сэмпл.
3. Одинаковая агрегация пользовательского профиля.

Поддерживаем два режима ранжирования:

1. `similarity`
   - профиль пользователя: средний вектор по истории;
   - score кандидата: `dot/cosine` с профилем.
2. `svm`
   - обучаем `LinearSVC` на train impressions;
   - признак пары `(user, item)`: `item_vector - user_profile_vector`;
   - score кандидата: `decision_function` модели.

Метрики качества:

1. `MRR`
2. `NDCG@5`, `NDCG@10`
3. `Recall@5`, `Recall@10`

Метрики времени:

1. `feature_time_sec`: время подготовки признаков за весь прогон.
2. `train_time_sec` (для SVM): построение train выборки + обучение.
3. `eval_time_sec`: время ранжирования за все impressions.
4. `total_time_sec`: сумма.

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

## SVM-результаты (LinearSVC, свежий прогон)

Конфигурация прогона:

1. `ranker=svm`
2. `max-impressions=300` (dev)
3. `svm-train-impressions=3000` (train)
4. `svm-max-samples=60000`
5. `model=nomic-embed-text:latest`

Результат (`TF-IDF+SVM` vs `Embeddings+SVM`):

1. `MRR`: `0.3282 -> 0.3513` (`+7.0%`)
2. `NDCG@10`: `0.3627 -> 0.3835` (`+5.7%`)
3. `Recall@10`: `0.5862 -> 0.6089` (`+3.9%`)

Времена того же прогона:

1. `TF-IDF+SVM total_time_sec`: `32.048s`
2. `Embeddings+SVM total_time_sec`: `80.805s`
3. Где узкое место для embeddings: `feature_time_sec` (построение эмбеддингов), а не `eval_time_sec`.

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

Для воспроизводимости используйте замороженные зависимости из lock-файла.

```bash
cd ~/code/mine/recsys-text-benchmark
uv sync --frozen
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

Только TF-IDF + similarity baseline:

```bash
uv run --frozen recsys-bench --mode tfidf --ranker similarity --max-impressions 3000 --seed 42
```

Только TF-IDF + SVM:

```bash
uv run --frozen recsys-bench --mode tfidf --ranker svm --max-impressions 300 --seed 42 \
  --svm-train-impressions 3000 --svm-max-samples 60000
```

Embeddings + SVM через локальный Ollama:

```bash
uv run --frozen recsys-bench --mode embeddings --ranker svm --max-impressions 300 --seed 42 \
  --svm-train-impressions 3000 --svm-max-samples 60000 \
  --ollama-model nomic-embed-text:latest --ollama-base-url http://127.0.0.1:11434
```

Embeddings + SVM через Docker Ollama:

```bash
uv run --frozen recsys-bench --mode embeddings --ranker svm --max-impressions 300 --seed 42 \
  --svm-train-impressions 3000 --svm-max-samples 60000 \
  --ollama-model nomic-embed-text:latest --ollama-base-url http://127.0.0.1:11435
```

Оба типа фич + SVM за один запуск:

```bash
uv run --frozen recsys-bench --mode both --ranker svm --max-impressions 300 --seed 42 \
  --svm-train-impressions 3000 --svm-max-samples 60000
```

## Что смотреть в выводе

1. `results/summary*.json`:
   - `dataset` (что именно было оценено),
   - `results.tfidf` / `results.embeddings` (similarity),
   - `results.tfidf_svm` / `results.embeddings_svm` (SVM),
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
