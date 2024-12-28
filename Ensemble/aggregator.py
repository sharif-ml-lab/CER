import re
from typing import List, Tuple, Dict, Optional


def extract_last_numerical_value(text: str) -> Optional[str]:
    """
    Extract the last numerical value from a given text.
    """
    matches = re.findall(r'\b\d+\.?\d*\b', text)
    return matches[-1] if matches else None


class AggregatorStrategy:
    """
    Abstract base class for aggregator strategies.
    """

    def aggregate(self, answers_with_probs):
        """
        :param answers_with_probs: A list of dicts:
            [
                {'answer': str, 'log_prob': float},
                ...
            ]
        :return: final ensembled answer as a string
        """
        raise NotImplementedError("Please implement an aggregator strategy.")


class MajorityVoteAggregator(AggregatorStrategy):
    """
    Simple majority vote aggregator based on the raw string of the answer.
    """

    def aggregate(self, answers_with_probs):
        # Tally answers
        tally = {}
        for ap in answers_with_probs:
            ans = ap['answer']
            final_ans = extract_last_numerical_value(ans)
            tally[final_ans] = tally.get(final_ans, 0) + 1

        # Pick the answer with the highest count
        final_answer = max(tally, key=tally.get)
        return final_answer


class MaxLogProbAggregator(AggregatorStrategy):
    """
    Selects the answer with the highest log probability.
    """

    def aggregate(self, answers_with_probs):
        # Sort by log_prob descending
        best = sorted(answers_with_probs, key=lambda x: x['log_prob'], reverse=True)[0]
        return best['answer']
