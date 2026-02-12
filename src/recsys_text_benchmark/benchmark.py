from __future__ import annotations

"""Бенчмарк текстовых фичей для рекомендательной задачи на MIND small.

Скрипт сравнивает два варианта представления текста:
1. TF-IDF.
2. Embeddings через Ollama.
"""

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

MIND_SMALL_TRAIN_URL = (
    "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip"
)
MIND_SMALL_DEV_URL = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip"


@dataclass
class RawImpression:
    """Сырые данные impression из MIND до привязки к индексам матриц."""

    history_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    labels: np.ndarray


@dataclass
class EvalImpression:
    """Impression после преобразования ID в индексы векторных матриц."""

    history_indices: np.ndarray
    candidate_indices: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    """Считать аргументы CLI для запуска эксперимента."""

    parser = argparse.ArgumentParser(
        description="Benchmark TF-IDF vs Ollama embeddings on MIND small."
    )
    parser.add_argument("--data-dir", default="data", help="Base directory for dataset.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where benchmark artifacts are written.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "tfidf", "embeddings"],
        help="What to run.",
    )
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=4000,
        help="How many dev impressions to evaluate (sampled).",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=50,
        help="How many recent history clicks to use per user profile.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--tfidf-max-features", type=int, default=50_000)
    parser.add_argument("--tfidf-max-ngram", type=int, default=2)
    parser.add_argument("--tfidf-min-df", type=int, default=2)

    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--ollama-batch-size", type=int, default=32)
    parser.add_argument("--ollama-timeout-seconds", type=int, default=600)

    parser.add_argument(
        "--output-json",
        default="results/summary.json",
        help="Where to write final metrics JSON.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    """Подготовить безопасное имя для файлового пути/кэша."""

    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def download_file(url: str, dst_path: Path) -> None:
    """Скачать файл по URL, если он ещё не существует локально."""

    if dst_path.exists():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, dst_path.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        progress = tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"download:{dst_path.name}",
        )
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            progress.update(len(chunk))
        progress.close()


def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> None:
    """Распаковать архив один раз и поставить маркер `.extracted`."""

    marker = extract_dir / ".extracted"
    if marker.exists():
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def resolve_mind_split_dir(base_dir: Path) -> Path:
    """Найти директорию сплита, где лежат `news.tsv` и `behaviors.tsv`."""

    if (base_dir / "news.tsv").exists() and (base_dir / "behaviors.tsv").exists():
        return base_dir

    candidates = sorted(
        news_file.parent
        for news_file in base_dir.rglob("news.tsv")
        if (news_file.parent / "behaviors.tsv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"Не найдены news.tsv/behaviors.tsv в {base_dir}")
    return candidates[0]


def ensure_mind_small(data_dir: Path) -> tuple[Path, Path]:
    """Скачать и подготовить train/dev части MIND small."""

    raw_dir = data_dir / "raw"
    train_zip = raw_dir / "MINDsmall_train.zip"
    dev_zip = raw_dir / "MINDsmall_dev.zip"
    train_dir = raw_dir / "MINDsmall_train"
    dev_dir = raw_dir / "MINDsmall_dev"

    download_file(MIND_SMALL_TRAIN_URL, train_zip)
    download_file(MIND_SMALL_DEV_URL, dev_zip)
    extract_zip_if_needed(train_zip, train_dir)
    extract_zip_if_needed(dev_zip, dev_dir)
    return resolve_mind_split_dir(train_dir), resolve_mind_split_dir(dev_dir)


def load_raw_impressions(
    behaviors_path: Path,
    max_impressions: int | None,
    seed: int,
) -> list[RawImpression]:
    """Прочитать impressions и отфильтровать невалидные примеры.

    Оставляем только случаи с непустой историей и смешанными метками
    (есть и позитивы, и негативы).
    """

    impressions: list[RawImpression] = []
    with behaviors_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue

            history_ids = tuple(token for token in row[3].split() if token)
            if not history_ids:
                continue

            candidate_ids: list[str] = []
            labels: list[int] = []
            for token in row[4].split():
                if "-" not in token:
                    continue
                news_id, label_text = token.rsplit("-", 1)
                try:
                    label = int(label_text)
                except ValueError:
                    continue
                candidate_ids.append(news_id)
                labels.append(label)

            if not candidate_ids:
                continue

            labels_array = np.asarray(labels, dtype=np.int8)
            positives = int(labels_array.sum())
            if positives == 0 or positives == len(labels_array):
                continue

            impressions.append(
                RawImpression(
                    history_ids=history_ids,
                    candidate_ids=tuple(candidate_ids),
                    labels=labels_array,
                )
            )

    # Для воспроизводимости сэмплируем фиксированным seed.
    if max_impressions is not None and len(impressions) > max_impressions:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(impressions), size=max_impressions, replace=False))
        impressions = [impressions[int(index)] for index in selected]

    return impressions


def load_news_texts(
    news_paths: list[Path],
    needed_news_ids: set[str],
) -> dict[str, str]:
    """Собрать текст новости (title + abstract) для нужных ID."""

    texts: dict[str, str] = {}
    for news_path in news_paths:
        with news_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                news_id = row[0]
                if news_id not in needed_news_ids or news_id in texts:
                    continue
                title = row[3].strip()
                abstract = row[4].strip()
                combined = f"{title}. {abstract}".strip()
                if combined in {".", ""}:
                    combined = title or abstract
                if combined:
                    texts[news_id] = combined
    return texts


def build_eval_impressions(
    raw_impressions: list[RawImpression],
    news_to_index: dict[str, int],
    max_history: int,
) -> tuple[list[EvalImpression], int]:
    """Преобразовать impression к индексам в матрицах признаков."""

    eval_impressions: list[EvalImpression] = []
    dropped = 0

    for impression in raw_impressions:
        history_indices = [news_to_index[n] for n in impression.history_ids if n in news_to_index]
        # Используем только последние клики, чтобы ограничить "хвост" истории.
        if max_history > 0 and len(history_indices) > max_history:
            history_indices = history_indices[-max_history:]
        if not history_indices:
            dropped += 1
            continue

        candidate_indices: list[int] = []
        candidate_labels: list[int] = []
        for idx, candidate_id in enumerate(impression.candidate_ids):
            candidate_index = news_to_index.get(candidate_id)
            if candidate_index is None:
                continue
            candidate_indices.append(candidate_index)
            candidate_labels.append(int(impression.labels[idx]))

        if len(candidate_indices) < 2:
            dropped += 1
            continue

        labels_array = np.asarray(candidate_labels, dtype=np.int8)
        positives = int(labels_array.sum())
        if positives == 0 or positives == len(labels_array):
            dropped += 1
            continue

        eval_impressions.append(
            EvalImpression(
                history_indices=np.asarray(history_indices, dtype=np.int32),
                candidate_indices=np.asarray(candidate_indices, dtype=np.int32),
                labels=labels_array,
            )
        )

    return eval_impressions, dropped


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Посчитать ranking-метрики для одного impression."""

    # Сортируем кандидатов по убыванию score.
    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    positives = int(ranked_labels.sum())
    if positives <= 0:
        return {
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
        }

    positive_positions = np.flatnonzero(ranked_labels == 1)
    mrr = 1.0 / float(positive_positions[0] + 1) if positive_positions.size else 0.0

    metrics: dict[str, float] = {"mrr": mrr}
    for k in (5, 10):
        top_k = ranked_labels[:k].astype(np.float64)
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(top_k)))
        dcg = float(np.sum(top_k * discounts))

        ideal_len = min(k, positives)
        ideal_discounts = 1.0 / np.log2(np.arange(2, 2 + ideal_len))
        idcg = float(np.sum(ideal_discounts))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recall = float(np.sum(top_k) / positives) if positives > 0 else 0.0

        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"recall@{k}"] = recall
    return metrics


def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """Усреднить метрики по всем impressions."""

    if not items:
        return {
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
        }
    keys = list(items[0].keys())
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def evaluate_with_tfidf(
    tfidf_matrix: sparse.csr_matrix,
    eval_impressions: list[EvalImpression],
) -> tuple[dict[str, float], float, int]:
    """Оценить качество и скорость ранжирования с TF-IDF признаками."""

    metrics: list[dict[str, float]] = []
    start = time.perf_counter()

    for impression in tqdm(eval_impressions, desc="eval:tfidf"):
        history = impression.history_indices
        # Профиль пользователя: средний вектор по истории кликов.
        user_vector = tfidf_matrix[int(history[0])].copy()
        for idx in history[1:]:
            user_vector += tfidf_matrix[int(idx)]
        user_vector = user_vector.multiply(1.0 / len(history))

        norm_sq = float(user_vector.multiply(user_vector).sum())
        if norm_sq <= 0:
            continue
        user_vector = user_vector.multiply(1.0 / math.sqrt(norm_sq))

        candidate_matrix = tfidf_matrix[impression.candidate_indices]
        scores = (candidate_matrix @ user_vector.T).toarray().ravel()
        metrics.append(compute_metrics(scores=scores, labels=impression.labels))

    elapsed = time.perf_counter() - start
    return aggregate_metrics(metrics), elapsed, len(metrics)


def evaluate_with_embeddings(
    embeddings: np.ndarray,
    eval_impressions: list[EvalImpression],
) -> tuple[dict[str, float], float, int]:
    """Оценить качество и скорость ранжирования с embedding-признаками."""

    metrics: list[dict[str, float]] = []
    start = time.perf_counter()

    for impression in tqdm(eval_impressions, desc="eval:embeddings"):
        history_vectors = embeddings[impression.history_indices]
        user_vector = history_vectors.mean(axis=0)
        norm = float(np.linalg.norm(user_vector))
        if norm <= 0:
            continue
        user_vector = user_vector / norm

        candidate_vectors = embeddings[impression.candidate_indices]
        scores = candidate_vectors @ user_vector
        metrics.append(compute_metrics(scores=scores, labels=impression.labels))

    elapsed = time.perf_counter() - start
    return aggregate_metrics(metrics), elapsed, len(metrics)


class OllamaClient:
    """Минимальный HTTP-клиент для Ollama embed API."""

    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        """Сохранить базовый URL и таймаут запросов."""

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Выполнить JSON-запрос к Ollama и вернуть JSON-ответ."""

        url = f"{self.base_url}{path}"
        data: bytes | None = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url=url, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            response_json = json.loads(raw)
            if not isinstance(response_json, dict):
                raise TypeError("Ollama API вернул неожиданный JSON-формат.")
            return cast(dict[str, Any], response_json)

    def check_health(self) -> None:
        """Проверить доступность Ollama перед массовыми embed-запросами."""

        try:
            self._request("/api/tags")
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "Ollama API недоступен. Запусти `ollama serve` и убедись, что модель скачана."
            ) from error

    def embed_batch(self, model: str, texts: list[str]) -> np.ndarray:
        """Получить эмбеддинги для батча текстов.

        При отсутствии `/api/embed` используется fallback на `/api/embeddings`.
        """

        try:
            response = self._request("/api/embed", {"model": model, "input": texts})
            embeddings = response.get("embeddings")
            if embeddings is None:
                raise RuntimeError("Ollama /api/embed вернул ответ без поля embeddings.")
            return np.asarray(embeddings, dtype=np.float32)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise

        vectors: list[list[float]] = []
        for text in texts:
            response = self._request("/api/embeddings", {"model": model, "prompt": text})
            embedding = response.get("embedding")
            if embedding is None:
                raise RuntimeError("Ollama /api/embeddings вернул ответ без поля embedding.")
            vectors.append(embedding)
        return np.asarray(vectors, dtype=np.float32)


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-нормировать каждую строку матрицы векторов."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def build_embedding_matrix(
    news_ids: list[str],
    texts: list[str],
    cache_dir: Path,
    model: str,
    base_url: str,
    batch_size: int,
    timeout_seconds: int,
) -> tuple[np.ndarray, Path, bool]:
    """Построить матрицу эмбеддингов и сохранить/использовать кэш."""

    # Ключ кэша зависит от модели и списка news_id (для воспроизводимости).
    key_src = f"{model}\n" + "\n".join(news_ids)
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]
    cache_name = f"embeddings_{sanitize_name(model)}_{len(news_ids)}_{key}.npz"
    cache_path = cache_dir / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    news_ids_np = np.asarray(news_ids, dtype=np.str_)

    if cache_path.exists():
        cached = np.load(cache_path)
        cached_news_ids = cached["news_ids"]
        if cached_news_ids.shape == news_ids_np.shape and np.array_equal(
            cached_news_ids, news_ids_np
        ):
            return cached["matrix"].astype(np.float32), cache_path, True

    client = OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds)
    client.check_health()

    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embed:ollama"):
        batch = texts[start : start + batch_size]
        embedded = client.embed_batch(model=model, texts=batch)
        chunks.append(embedded)

    matrix = np.vstack(chunks).astype(np.float32)
    matrix = l2_normalize_rows(matrix)
    np.savez_compressed(cache_path, news_ids=news_ids_np, matrix=matrix)
    return matrix, cache_path, False


def print_results(results: dict[str, dict[str, Any]]) -> None:
    """Вывести сводные результаты в удобочитаемом виде."""

    for name, payload in results.items():
        metrics = payload["metrics"]
        print(f"\n[{name}]")
        print(f"  feature_time_sec: {payload['feature_time_sec']:.3f}")
        print(f"  eval_time_sec:    {payload['eval_time_sec']:.3f}")
        print(f"  total_time_sec:   {payload['total_time_sec']:.3f}")
        print(f"  evaluated_impr:   {payload['evaluated_impressions']}")
        print(
            "  metrics:"
            f" mrr={metrics['mrr']:.4f}"
            f" ndcg@5={metrics['ndcg@5']:.4f}"
            f" ndcg@10={metrics['ndcg@10']:.4f}"
            f" recall@5={metrics['recall@5']:.4f}"
            f" recall@10={metrics['recall@10']:.4f}"
        )


def main() -> None:
    """Точка входа CLI: подготовка данных, запуск оценки, сохранение отчёта."""

    args = parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dir, dev_dir = ensure_mind_small(data_dir)
    raw_impressions = load_raw_impressions(
        behaviors_path=dev_dir / "behaviors.tsv",
        max_impressions=args.max_impressions,
        seed=args.seed,
    )
    if not raw_impressions:
        raise RuntimeError("Не удалось подготовить impressions для оценки.")

    # Ограничиваем корпус только новостями, которые реально встречаются в выборке.
    needed_ids: set[str] = set()
    for impression in raw_impressions:
        needed_ids.update(impression.history_ids)
        needed_ids.update(impression.candidate_ids)

    news_texts = load_news_texts(
        news_paths=[train_dir / "news.tsv", dev_dir / "news.tsv"],
        needed_news_ids=needed_ids,
    )
    if not news_texts:
        raise RuntimeError("Не удалось загрузить тексты новостей.")

    news_ids = sorted(news_texts.keys())
    text_corpus = [news_texts[news_id] for news_id in news_ids]
    news_to_index = {news_id: idx for idx, news_id in enumerate(news_ids)}

    eval_impressions, dropped = build_eval_impressions(
        raw_impressions=raw_impressions,
        news_to_index=news_to_index,
        max_history=args.max_history,
    )
    if not eval_impressions:
        raise RuntimeError("После фильтрации не осталось impressions для оценки.")

    print(
        "dataset:"
        f" raw_impressions={len(raw_impressions)}"
        f" eval_impressions={len(eval_impressions)}"
        f" dropped={dropped}"
        f" unique_news={len(news_ids)}"
    )

    summary: dict[str, Any] = {
        "dataset": {
            "name": "MINDsmall_dev",
            "raw_impressions": len(raw_impressions),
            "eval_impressions": len(eval_impressions),
            "dropped_impressions": dropped,
            "unique_news": len(news_ids),
            "max_history": args.max_history,
            "max_impressions": args.max_impressions,
            "seed": args.seed,
        },
        "results": {},
    }

    results: dict[str, dict[str, Any]] = {}

    if args.mode in {"both", "tfidf"}:
        # Отдельно меряем стоимость подготовки TF-IDF признаков.
        feature_start = time.perf_counter()
        vectorizer = TfidfVectorizer(
            max_features=args.tfidf_max_features,
            ngram_range=(1, args.tfidf_max_ngram),
            min_df=args.tfidf_min_df,
            strip_accents="unicode",
        )
        tfidf_matrix = vectorizer.fit_transform(text_corpus).tocsr().astype(np.float32)
        feature_time = time.perf_counter() - feature_start

        metrics, eval_time, evaluated = evaluate_with_tfidf(
            tfidf_matrix=tfidf_matrix,
            eval_impressions=eval_impressions,
        )
        results["tfidf"] = {
            "feature_time_sec": feature_time,
            "eval_time_sec": eval_time,
            "total_time_sec": feature_time + eval_time,
            "evaluated_impressions": evaluated,
            "vocab_size": len(vectorizer.vocabulary_),
            "metrics": metrics,
        }
        summary["results"]["tfidf"] = results["tfidf"]

    if args.mode in {"both", "embeddings"}:
        # Отдельно меряем стоимость построения embeddings (cold/warm по кэшу).
        feature_start = time.perf_counter()
        embeddings, cache_path, loaded_from_cache = build_embedding_matrix(
            news_ids=news_ids,
            texts=text_corpus,
            cache_dir=data_dir / "processed",
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            batch_size=args.ollama_batch_size,
            timeout_seconds=args.ollama_timeout_seconds,
        )
        feature_time = time.perf_counter() - feature_start

        metrics, eval_time, evaluated = evaluate_with_embeddings(
            embeddings=embeddings,
            eval_impressions=eval_impressions,
        )
        results["embeddings"] = {
            "feature_time_sec": feature_time,
            "eval_time_sec": eval_time,
            "total_time_sec": feature_time + eval_time,
            "evaluated_impressions": evaluated,
            "embedding_model": args.ollama_model,
            "embedding_dim": int(embeddings.shape[1]),
            "cache_path": str(cache_path),
            "loaded_from_cache": loaded_from_cache,
            "metrics": metrics,
        }
        summary["results"]["embeddings"] = results["embeddings"]

    print_results(results)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten: {output_json}")


if __name__ == "__main__":
    main()
