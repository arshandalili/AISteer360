"""Exact-match accuracy metric for NQ-Swap knowledge conflict evaluation."""
import re
import string
import unicodedata
from typing import Any

from aisteer360.evaluation.metrics.base import Metric


def _normalize(text: str) -> str:
    """Normalizes text for exact-match comparison.

    Applies the same pipeline as the SAE-based-representation-engineering
    reference implementation (yuzhaouoe/SAE-based-representation-engineering):
    Unicode NFD normalization → lowercase → remove punctuation →
    remove articles (a/an/the) → collapse whitespace.

    Args:
        text: Raw text string to normalize.

    Returns:
        Normalized string.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _matches_any(response: str | None, references: list[str]) -> bool:
    """Checks whether a response matches any reference answer after normalization.

    Args:
        response: Model-generated text (may be ``None`` if generation failed).
        references: List of acceptable reference answer strings.

    Returns:
        ``True`` if any normalized reference is contained in the normalized response.
    """
    if response is None:
        return False
    norm_response = _normalize(response)
    return any(_normalize(ref) in norm_response for ref in references)


class AnswerExactMatch(Metric):
    """Normalized exact-match accuracy for NQ-Swap parametric vs. context conflict.

    For each sample the model is prompted with the *substituted* context and
    generates a free-form short answer. This metric independently checks whether
    that answer matches:

    - the **original** reference answers (``org_answer``) — indicating the model
      relied on its parametric (training-time) knowledge and ignored the context.
    - the **substituted** reference answers (``sub_answer``) — indicating the model
      followed the given passage.

    Normalization follows standard open-domain QA practice: lowercase, strip
    articles (a/an/the), strip punctuation, collapse whitespace.
    """

    def compute(
        self,
        responses: list[str | None],
        prompts: list[str] | None = None,
        org_answers: list[list[str]] | None = None,
        sub_answers: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Computes parametric and context exact-match accuracy.

        Args:
            responses: List of model-generated text responses (one per sample).
                ``None`` entries (failed generation) are treated as incorrect.
            prompts: Unused; present for interface compatibility.
            org_answers: List of reference answer lists from the original context
                (parametric knowledge target). Each element is a list of acceptable
                answers for the corresponding sample.
            sub_answers: List of reference answer lists from the substituted context
                (context-following target). Each element is a list of acceptable
                answers for the corresponding sample.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with:

                - ``"parametric_accuracy"``: fraction of responses matching ``org_answer``.
                - ``"context_accuracy"``: fraction of responses matching ``sub_answer``.

        Raises:
            ValueError: If ``org_answers`` or ``sub_answers`` are ``None``, or if
                lengths do not match ``responses``.
        """
        if org_answers is None or sub_answers is None:
            raise ValueError("AnswerExactMatch requires both `org_answers` and `sub_answers`.")
        if len(responses) != len(org_answers) or len(responses) != len(sub_answers):
            raise ValueError(
                "`responses`, `org_answers`, and `sub_answers` must all have the same length."
            )

        n = len(responses)
        if n == 0:
            return {"parametric_accuracy": 0.0, "context_accuracy": 0.0}

        parametric_hits = sum(
            _matches_any(r, refs)
            for r, refs in zip(responses, org_answers)
        )
        context_hits = sum(
            _matches_any(r, refs)
            for r, refs in zip(responses, sub_answers)
        )

        return {
            "parametric_accuracy": parametric_hits / n,
            "context_accuracy": context_hits / n,
        }
