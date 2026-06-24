from __future__ import annotations

from typing import Dict, Sequence, Set


DEFAULT_CUTOFFS = (1, 10, 20, 30, 40, 50)


class RankingMetrics:
    """Query-level ranking metrics for one-to-one and one-to-many mappings.

    Hit@K measures whether Top-K contains any positive target. It is not the
    strict IR recall over all relevant targets; use AllTargetRecall@K for that.
    """

    def __init__(self, cutoffs: Sequence[int] = DEFAULT_CUTOFFS):
        self.cutoffs = tuple(dict.fromkeys(cutoffs))
        self.queries = 0
        self.hit_counts = {k: 0 for k in self.cutoffs}
        self.all_target_recall = {k: 0.0 for k in self.cutoffs}
        self.reciprocal_rank = 0.0

    def add(self, ranked_uids: Sequence[str], positives: Set[str]) -> None:
        self.queries += 1
        if not positives:
            return

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
            self.all_target_recall[k] += len(top & positives) / len(positives)

    def result(self) -> Dict[str, float]:
        denominator = max(self.queries, 1)
        result = {
            f"Hit@{k}": self.hit_counts[k] / denominator
            for k in self.cutoffs
        }
        result.update({
            f"Acc@{k}": result[f"Hit@{k}"]
            for k in self.cutoffs
        })
        result.update({
            f"AllTargetRecall@{k}": self.all_target_recall[k] / denominator
            for k in self.cutoffs
        })
        result["MRR"] = self.reciprocal_rank / denominator
        result["queries"] = self.queries
        return result
