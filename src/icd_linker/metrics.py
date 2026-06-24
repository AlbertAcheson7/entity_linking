from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set


class RankingMetrics:
    def __init__(self, cutoffs: Sequence[int] = (1, 5, 10, 50)):
        self.cutoffs = tuple(sorted(cutoffs))
        self.queries = 0
        self.hit_counts = {k: 0 for k in self.cutoffs}
        self.all_target_recall = {k: 0.0 for k in self.cutoffs}
        self.reciprocal_rank = 0.0

    def add(self, ranked_uids: Sequence[str], positives: Set[str]) -> None:
        self.queries += 1
        first_rank = None
        for rank, uid in enumerate(ranked_uids, 1):
            if uid in positives:
                first_rank = rank
                break
        if first_rank:
            self.reciprocal_rank += 1.0 / first_rank
        for k in self.cutoffs:
            top = set(ranked_uids[:k])
            self.hit_counts[k] += bool(top & positives)
            self.all_target_recall[k] += (
                len(top & positives) / len(positives) if positives else 0.0
            )

    def result(self) -> Dict[str, float]:
        denominator = max(self.queries, 1)
        result = {
            f"Recall@{k}": self.hit_counts[k] / denominator
            for k in self.cutoffs
        }
        result.update({
            f"AllTargetRecall@{k}": self.all_target_recall[k] / denominator
            for k in self.cutoffs
        })
        result["MRR"] = self.reciprocal_rank / denominator
        result["queries"] = self.queries
        return result

