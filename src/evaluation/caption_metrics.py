"""Metric computation for Stage 2 Open-Ended WSI Captioning & Pathology Report Generation.

Provides self-contained implementations of standard image-captioning and clinical NLG metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (corpus-level Papineni BLEU and smoothed sentence-level BLEU)
- ROUGE-1, ROUGE-2, ROUGE-L (F1)
- METEOR (unigram stem/exact alignment with fragmentation penalty)
- CIDEr-D (TF-IDF weighted 1..4-gram cosine similarity with Gaussian length penalty)
- BERTScore (optional contextual embedding P/R/F1 via `transformers` or `bert_score`)
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\+[a-z0-9]*)?")


def tokenize_report(text: str) -> List[str]:
    """Tokenize a clinical pathology report into normalized lowercase words.

    Preserves alphanumeric clinical markers (e.g., 'her2+', 'g3', 'ck14', 't2n0m0').

    Args:
        text: Raw caption or pathology report string.

    Returns:
        List of lowercase word tokens.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _extract_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Return a Counter of contiguous n-grams from `tokens`."""
    num_tokens = len(tokens)
    if num_tokens < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(num_tokens - n + 1))


def _light_stem(word: str) -> str:
    """Apply lightweight morphological suffix stripping for METEOR stem matching."""
    if len(word) <= 4:
        return word
    for suffix in (
        "carcinomas",
        "adenocarcinomas",
        "ations",
        "ation",
        "ities",
        "ously",
        "ments",
        "ment",
        "ness",
        "ings",
        "ing",
        "ally",
        "edly",
        "ular",
        "ous",
        "ive",
        "ies",
        "es",
        "ed",
        "ly",
        "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix in ("carcinomas", "adenocarcinomas"):
                return word[:-1]
            if suffix == "ies":
                return word[:-3] + "y"
            return word[: -len(suffix)]
    return word


def compute_sentence_bleu(
    pred_tokens: Sequence[str],
    ref_tokens: Sequence[str],
    max_n: int = 4,
    smooth_eps: float = 0.1,
) -> Dict[str, float]:
    """Compute smoothed sentence-level BLEU-1..4 (Chen & Cherry method 1).

    Args:
        pred_tokens: Tokenized candidate report.
        ref_tokens: Tokenized ground-truth reference report.
        max_n: Maximum n-gram order (default: 4).
        smooth_eps: Add-epsilon smoothing for zero higher-order n-gram counts.

    Returns:
        Dictionary mapping `'bleu_1'..'bleu_4'` to scores in `[0.0, 1.0]`.
    """
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0 or ref_len == 0:
        return {f"bleu_{k}": 0.0 for k in range(1, max_n + 1)}

    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / pred_len))

    log_precisions: List[float] = []
    scores: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        pred_ngrams = _extract_ngrams(pred_tokens, n)
        ref_ngrams = _extract_ngrams(ref_tokens, n)
        total = sum(pred_ngrams.values())
        if total == 0:
            p_n = 0.0
        else:
            clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items())
            if clipped == 0:
                p_n = smooth_eps / total if n > 1 else 0.0
            else:
                p_n = clipped / total

        if p_n <= 0.0:
            log_precisions.append(float("-inf"))
        else:
            log_precisions.append(math.log(p_n))

        if any(math.isinf(lp) for lp in log_precisions):
            scores[f"bleu_{n}"] = 0.0
        else:
            mean_log = sum(log_precisions) / n
            scores[f"bleu_{n}"] = bp * math.exp(mean_log)

    return scores


def compute_corpus_bleu(
    tokenized_preds: Sequence[Sequence[str]],
    tokenized_refs: Sequence[Sequence[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute exact corpus-level Papineni BLEU-1..4 across a list of predictions and references.

    Args:
        tokenized_preds: Sequence of tokenized candidate reports.
        tokenized_refs: Sequence of tokenized ground-truth reference reports.
        max_n: Maximum n-gram order (default: 4).

    Returns:
        Dictionary mapping `'corpus_bleu_1'..'corpus_bleu_4'` to scores in `[0.0, 1.0]`.
    """
    clipped_hits = [0] * max_n
    total_counts = [0] * max_n
    pred_total_len = 0
    ref_total_len = 0

    for pred_tokens, ref_tokens in zip(tokenized_preds, tokenized_refs):
        pred_total_len += len(pred_tokens)
        ref_total_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            p_ng = _extract_ngrams(pred_tokens, n)
            r_ng = _extract_ngrams(ref_tokens, n)
            total_counts[n - 1] += sum(p_ng.values())
            clipped_hits[n - 1] += sum(min(c, r_ng[g]) for g, c in p_ng.items())

    if pred_total_len == 0 or ref_total_len == 0:
        return {f"corpus_bleu_{k}": 0.0 for k in range(1, max_n + 1)}

    bp = (
        1.0
        if pred_total_len > ref_total_len
        else math.exp(1.0 - (ref_total_len / pred_total_len))
    )

    log_precs: List[float] = []
    results: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        tot = total_counts[n - 1]
        hits = clipped_hits[n - 1]
        if tot == 0 or hits == 0:
            log_precs.append(float("-inf"))
        else:
            log_precs.append(math.log(hits / tot))

        if any(math.isinf(lp) for lp in log_precs):
            results[f"corpus_bleu_{n}"] = 0.0
        else:
            results[f"corpus_bleu_{n}"] = bp * math.exp(sum(log_precs) / n)

    return results


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute exact Longest Common Subsequence length in O(min(N, M)) space."""
    if not a or not b:
        return 0
    # Trim shared prefix and suffix to accelerate long reports
    start = 0
    min_len = min(len(a), len(b))
    while start < min_len and a[start] == b[start]:
        start += 1
    end_a = len(a)
    end_b = len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1

    shared_edges = start + (len(a) - end_a)
    sub_a = a[start:end_a]
    sub_b = b[start:end_b]
    if not sub_a or not sub_b:
        return shared_edges

    if len(sub_a) < len(sub_b):
        short_seq, long_seq = sub_a, sub_b
    else:
        short_seq, long_seq = sub_b, sub_a

    dp = [0] * (len(short_seq) + 1)
    for token_l in long_seq:
        prev_diag = 0
        for j, token_s in enumerate(short_seq, start=1):
            temp = dp[j]
            if token_l == token_s:
                dp[j] = prev_diag + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev_diag = temp

    return shared_edges + dp[-1]


def compute_rouge_scores(
    pred_tokens: Sequence[str], ref_tokens: Sequence[str]
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for a single prediction/reference pair.

    Args:
        pred_tokens: Tokenized candidate report.
        ref_tokens: Tokenized ground-truth reference report.

    Returns:
        Dictionary with `'rouge_1'`, `'rouge_2'`, and `'rouge_l'` F1 scores in `[0.0, 1.0]`.
    """
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0 or ref_len == 0:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    scores: Dict[str, float] = {}
    for n, key in ((1, "rouge_1"), (2, "rouge_2")):
        p_ng = _extract_ngrams(pred_tokens, n)
        r_ng = _extract_ngrams(ref_tokens, n)
        p_tot = sum(p_ng.values())
        r_tot = sum(r_ng.values())
        if p_tot == 0 or r_tot == 0:
            scores[key] = 0.0
        else:
            overlap = sum(min(c, r_ng[g]) for g, c in p_ng.items())
            prec = overlap / p_tot
            rec = overlap / r_tot
            scores[key] = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec_l = lcs / pred_len
    rec_l = lcs / ref_len
    scores["rouge_l"] = (
        (2.0 * prec_l * rec_l / (prec_l + rec_l)) if (prec_l + rec_l) > 0 else 0.0
    )
    return scores


def compute_meteor_score(
    pred_tokens: Sequence[str], ref_tokens: Sequence[str]
) -> float:
    """Compute single-pair METEOR score with exact + stem alignment and chunk penalty.

    Follows Banerjee & Lavie (2005) parameterization (`alpha=0.9`, `beta=3.0`, `gamma=0.5`).

    Args:
        pred_tokens: Tokenized candidate report.
        ref_tokens: Tokenized ground-truth reference report.

    Returns:
        METEOR score in `[0.0, 1.0]`.
    """
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0 or ref_len == 0:
        return 0.0

    ref_used = [False] * ref_len
    alignment: List[Tuple[int, int]] = []

    # Stage 1: Exact unigram match
    ref_positions_by_word: Dict[str, List[int]] = {}
    for j, tok in enumerate(ref_tokens):
        ref_positions_by_word.setdefault(tok, []).append(j)

    pred_matched = [False] * pred_len
    for i, tok in enumerate(pred_tokens):
        for j in ref_positions_by_word.get(tok, ()):
            if not ref_used[j]:
                ref_used[j] = True
                pred_matched[i] = True
                alignment.append((i, j))
                break

    # Stage 2: Morphological stem match for remaining unmatched tokens
    ref_positions_by_stem: Dict[str, List[int]] = {}
    for j, tok in enumerate(ref_tokens):
        if not ref_used[j]:
            ref_positions_by_stem.setdefault(_light_stem(tok), []).append(j)

    for i, tok in enumerate(pred_tokens):
        if pred_matched[i]:
            continue
        stem = _light_stem(tok)
        for j in ref_positions_by_stem.get(stem, ()):
            if not ref_used[j]:
                ref_used[j] = True
                pred_matched[i] = True
                alignment.append((i, j))
                break

    matches = len(alignment)
    if matches == 0:
        return 0.0

    alignment.sort(key=lambda pair: pair[0])
    chunks = 1
    for idx in range(1, matches):
        prev_i, prev_j = alignment[idx - 1]
        curr_i, curr_j = alignment[idx]
        if curr_i != prev_i + 1 or curr_j != prev_j + 1:
            chunks += 1

    precision = matches / pred_len
    recall = matches / ref_len
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)
    frag = chunks / matches
    penalty = 0.5 * (frag ** 3)
    return float(f_mean * (1.0 - penalty))


def compute_corpus_cider(
    tokenized_preds: Sequence[Sequence[str]],
    tokenized_refs: Sequence[Sequence[str]],
    max_n: int = 4,
    sigma: float = 6.0,
) -> Tuple[float, List[float]]:
    """Compute CIDEr-D across a corpus of candidate and reference reports (Vedantam et al., CVPR 2015).

    Args:
        tokenized_preds: Sequence of tokenized candidate reports.
        tokenized_refs: Sequence of tokenized ground-truth reference reports.
        max_n: Maximum n-gram order (default: 4).
        sigma: Gaussian length-penalty standard deviation (default: 6.0).

    Returns:
        Tuple of `(mean_cider_score, per_sample_cider_scores)` scaled by 10.0 (standard CIDEr scale).
    """
    num_samples = len(tokenized_preds)
    if num_samples == 0:
        return 0.0, []

    ref_len_log = math.log(max(1.0, float(num_samples)))
    doc_freqs: List[Counter] = [Counter() for _ in range(max_n)]
    pred_ngram_bags: List[List[Counter]] = []
    ref_ngram_bags: List[List[Counter]] = []

    for pred_toks, ref_toks in zip(tokenized_preds, tokenized_refs):
        p_bags: List[Counter] = []
        r_bags: List[Counter] = []
        for n in range(1, max_n + 1):
            p_c = _extract_ngrams(pred_toks, n)
            r_c = _extract_ngrams(ref_toks, n)
            p_bags.append(p_c)
            r_bags.append(r_c)
            for ng in r_c.keys():
                doc_freqs[n - 1][ng] += 1
        pred_ngram_bags.append(p_bags)
        ref_ngram_bags.append(r_bags)

    per_sample_scores: List[float] = []
    for idx in range(num_samples):
        pred_toks = tokenized_preds[idx]
        ref_toks = tokenized_refs[idx]
        if not pred_toks or not ref_toks:
            per_sample_scores.append(0.0)
            continue

        delta_len = len(pred_toks) - len(ref_toks)
        len_penalty = math.exp(-(delta_len ** 2) / (2.0 * (sigma ** 2)))

        n_sims: List[float] = []
        for n_idx in range(max_n):
            p_c = pred_ngram_bags[idx][n_idx]
            r_c = ref_ngram_bags[idx][n_idx]
            p_total = sum(p_c.values())
            r_total = sum(r_c.values())
            if p_total == 0 or r_total == 0:
                n_sims.append(0.0)
                continue

            df_n = doc_freqs[n_idx]
            p_weights: Dict[Tuple[str, ...], float] = {}
            r_weights: Dict[Tuple[str, ...], float] = {}

            for ng, cnt in p_c.items():
                idf = max(0.0, ref_len_log - math.log(max(1, df_n.get(ng, 1))))
                # CIDEr-D clips candidate count at reference count in the numerator
                clipped_cnt = min(cnt, r_c.get(ng, 0))
                p_weights[ng] = (clipped_cnt / p_total) * idf

            p_norm_sq = 0.0
            for ng, cnt in p_c.items():
                idf = max(0.0, ref_len_log - math.log(max(1, df_n.get(ng, 1))))
                val = (cnt / p_total) * idf
                p_norm_sq += val * val

            r_norm_sq = 0.0
            for ng, cnt in r_c.items():
                idf = max(0.0, ref_len_log - math.log(max(1, df_n.get(ng, 1))))
                val = (cnt / r_total) * idf
                r_weights[ng] = val
                r_norm_sq += val * val

            denom = math.sqrt(p_norm_sq) * math.sqrt(r_norm_sq)
            if denom <= 1e-12:
                n_sims.append(0.0)
            else:
                dot = sum(
                    p_weights[ng] * r_weights[ng]
                    for ng in p_weights.keys()
                    if ng in r_weights
                )
                n_sims.append((dot / denom) * len_penalty)

        cider_i = 10.0 * (sum(n_sims) / max_n)
        per_sample_scores.append(cider_i)

    mean_cider = sum(per_sample_scores) / num_samples
    return mean_cider, per_sample_scores


def compute_bertscore_batch(
    predictions: Sequence[str],
    references: Sequence[str],
    model_name_or_path: str = "roberta-large",
    device: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> Optional[Dict[str, Any]]:
    """Compute contextual BERTScore (Precision, Recall, F1) using `transformers` + `torch`.

    Implements exact greedy cosine-similarity matching over contextual token embeddings
    (excluding special BOS/EOS/PAD tokens) so no external `bert-score` pip package is required.

    Args:
        predictions: Sequence of generated pathology report strings.
        references: Sequence of ground-truth pathology report strings.
        model_name_or_path: HuggingFace model ID or local directory path.
        device: Target device string (e.g., `'cuda:0'` or `'cpu'`).
        batch_size: Mini-batch size for transformer encoding.
        max_length: Maximum token sequence length.

    Returns:
        Dictionary containing `'precision'`, `'recall'`, `'f1'` (macro averages) and
        `'per_sample_f1'`, or `None` if `torch`/`transformers` or the model weights are unavailable.
    """
    if not predictions or not references:
        return None

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path).to(resolved_device)
        model.eval()
    except Exception:
        return None

    special_ids = set(tokenizer.all_special_ids)
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            batch_preds = [
                p.strip() if p and p.strip() else "."
                for p in predictions[start : start + batch_size]
            ]
            batch_refs = [
                r.strip() if r and r.strip() else "."
                for r in references[start : start + batch_size]
            ]

            enc_p = tokenizer(
                batch_preds,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(resolved_device)
            enc_r = tokenizer(
                batch_refs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(resolved_device)

            emb_p = F.normalize(model(**enc_p).last_hidden_state, p=2, dim=-1)
            emb_r = F.normalize(model(**enc_r).last_hidden_state, p=2, dim=-1)

            for i in range(len(batch_preds)):
                raw_pred = predictions[start + i]
                raw_ref = references[start + i]
                if not raw_pred or not raw_pred.strip() or not raw_ref or not raw_ref.strip():
                    precisions.append(0.0)
                    recalls.append(0.0)
                    f1s.append(0.0)
                    continue

                p_ids = enc_p["input_ids"][i].tolist()
                r_ids = enc_r["input_ids"][i].tolist()
                p_valid = [idx for idx, tid in enumerate(p_ids) if tid not in special_ids]
                r_valid = [idx for idx, tid in enumerate(r_ids) if tid not in special_ids]

                if not p_valid or not r_valid:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    f1s.append(0.0)
                    continue

                vecs_p = emb_p[i, p_valid]  # (Lp, D)
                vecs_r = emb_r[i, r_valid]  # (Lr, D)
                sim_matrix = torch.matmul(vecs_p, vecs_r.transpose(0, 1))  # (Lp, Lr)

                p_score = float(sim_matrix.max(dim=1).values.mean().item())
                r_score = float(sim_matrix.max(dim=0).values.mean().item())
                f1_score = (
                    (2.0 * p_score * r_score / (p_score + r_score))
                    if (p_score + r_score) > 0
                    else 0.0
                )
                precisions.append(p_score)
                recalls.append(r_score)
                f1s.append(f1_score)

    n = len(f1s)
    return {
        "model": model_name_or_path,
        "precision": round(sum(precisions) / n, 4),
        "recall": round(sum(recalls) / n, 4),
        "f1": round(sum(f1s) / n, 4),
        "per_sample_f1": [round(val, 4) for val in f1s],
    }


def compute_sample_lexical_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Compute per-sample lexical captioning metrics for incremental `.jsonl` logging.

    Args:
        prediction: Model-generated pathology report / caption string.
        reference: Ground-truth reference pathology report string.

    Returns:
        Dictionary with `bleu_1..4`, `rouge_1`, `rouge_2`, `rouge_l`, `meteor`,
        `pred_word_count`, and `ref_word_count`.
    """
    pred_tokens = tokenize_report(prediction)
    ref_tokens = tokenize_report(reference)

    bleu = compute_sentence_bleu(pred_tokens, ref_tokens)
    rouge = compute_rouge_scores(pred_tokens, ref_tokens)
    meteor = compute_meteor_score(pred_tokens, ref_tokens)

    return {
        "bleu_1": round(bleu["bleu_1"], 4),
        "bleu_2": round(bleu["bleu_2"], 4),
        "bleu_3": round(bleu["bleu_3"], 4),
        "bleu_4": round(bleu["bleu_4"], 4),
        "rouge_1": round(rouge["rouge_1"], 4),
        "rouge_2": round(rouge["rouge_2"], 4),
        "rouge_l": round(rouge["rouge_l"], 4),
        "meteor": round(meteor, 4),
        "pred_word_count": len(pred_tokens),
        "ref_word_count": len(ref_tokens),
    }


def _aggregate_group_records(group_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute corpus and macro-averaged caption metrics for a slice of prediction records."""
    count = len(group_records)
    if count == 0:
        return {"total_samples": 0}

    preds = [str(r.get("raw_output", "") or "") for r in group_records]
    refs = [
        str(r.get("reference_caption") or r.get("answer_text") or "")
        for r in group_records
    ]
    tok_preds = [tokenize_report(p) for p in preds]
    tok_refs = [tokenize_report(r) for r in refs]

    empty_count = sum(1 for t in tok_preds if len(t) == 0)
    corpus_bleu = compute_corpus_bleu(tok_preds, tok_refs)
    mean_cider, _ = compute_corpus_cider(tok_preds, tok_refs)

    sum_metrics = Counter()
    for idx, rec in enumerate(group_records):
        if "bleu_1" in rec and "rouge_l" in rec and "meteor" in rec:
            for k in (
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "meteor",
            ):
                sum_metrics[k] += float(rec.get(k, 0.0))
        else:
            s_metrics = compute_sample_lexical_metrics(preds[idx], refs[idx])
            for k in (
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "meteor",
            ):
                sum_metrics[k] += s_metrics[k]

    summary: Dict[str, Any] = {
        "total_samples": count,
        "empty_predictions": empty_count,
        "bleu_1": round(sum_metrics["bleu_1"] / count, 4),
        "bleu_2": round(sum_metrics["bleu_2"] / count, 4),
        "bleu_3": round(sum_metrics["bleu_3"] / count, 4),
        "bleu_4": round(sum_metrics["bleu_4"] / count, 4),
        "corpus_bleu_1": round(corpus_bleu["corpus_bleu_1"], 4),
        "corpus_bleu_2": round(corpus_bleu["corpus_bleu_2"], 4),
        "corpus_bleu_3": round(corpus_bleu["corpus_bleu_3"], 4),
        "corpus_bleu_4": round(corpus_bleu["corpus_bleu_4"], 4),
        "rouge_1": round(sum_metrics["rouge_1"] / count, 4),
        "rouge_2": round(sum_metrics["rouge_2"] / count, 4),
        "rouge_l": round(sum_metrics["rouge_l"] / count, 4),
        "meteor": round(sum_metrics["meteor"] / count, 4),
        "cider": round(mean_cider, 4),
        "avg_pred_words": round(sum(len(t) for t in tok_preds) / count, 1),
        "avg_ref_words": round(sum(len(t) for t in tok_refs) / count, 1),
    }

    bert_f1_vals = [
        float(r["bertscore_f1"])
        for r in group_records
        if r.get("bertscore_f1") is not None
    ]
    if bert_f1_vals:
        summary["bertscore_f1"] = round(sum(bert_f1_vals) / len(bert_f1_vals), 4)

    return summary


def compute_caption_metrics(
    records: Sequence[Dict[str, Any]],
    compute_bertscore: bool = False,
    bertscore_model: str = "roberta-large",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute overall, per-cohort, and per-task (Tumor/Organ) Stage 2 captioning metrics.

    Args:
        records: Sequence of evaluated caption prediction records.
        compute_bertscore: If True, also compute contextual BERTScore P/R/F1.
        bertscore_model: HuggingFace model identifier or local path for BERTScore.
        device: Optional device string for BERTScore computation.

    Returns:
        Dictionary containing overall metrics, `'by_cohort'`, and `'by_task'` breakdowns.
    """
    if not records:
        return {"total_samples": 0}

    if compute_bertscore:
        preds = [str(r.get("raw_output", "") or "") for r in records]
        refs = [
            str(r.get("reference_caption") or r.get("answer_text") or "")
            for r in records
        ]
        bert_res = compute_bertscore_batch(
            predictions=preds,
            references=refs,
            model_name_or_path=bertscore_model,
            device=device,
        )
        if bert_res is not None:
            for rec, f1_val in zip(records, bert_res["per_sample_f1"]):
                if isinstance(rec, dict):
                    rec["bertscore_f1"] = f1_val

    overall = _aggregate_group_records(records)

    by_cohort_records: Dict[str, List[Dict[str, Any]]] = {}
    by_task_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        cohort_key = str(rec.get("cohort", "unknown"))
        task_key = str(rec.get("task", "unknown"))
        by_cohort_records.setdefault(cohort_key, []).append(rec)
        by_task_records.setdefault(task_key, []).append(rec)

    overall["by_cohort"] = {
        k: _aggregate_group_records(v) for k, v in sorted(by_cohort_records.items())
    }
    overall["by_task"] = {
        k: _aggregate_group_records(v) for k, v in sorted(by_task_records.items())
    }
    return overall
