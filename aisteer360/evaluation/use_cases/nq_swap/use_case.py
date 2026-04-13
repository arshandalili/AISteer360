"""NQ-Swap use case for evaluating parametric vs. context knowledge conflict."""

import json
import math
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVALUATION_REQ_KEYS = [
    "question",
    "sub_context",
    "org_answer",
    "sub_answer",
]

_PROMPT_TEMPLATE = "context: {sub_context}\n" "question: {question}\n" "answer:"


class NQSwap(UseCase):
    """NQ-Swap knowledge conflict evaluation use case.

    Evaluates whether steering methods can guide a model to answer from its
    parametric memory or from a given (substituted) context passage. Each
    evaluation instance pairs a question with both an original passage
    (reflecting the model's training knowledge) and a substituted passage
    (with a swapped entity).

    The model is always prompted with the *substituted* context so that there
    is an explicit conflict between the passage content and the model's
    parametric knowledge. The ``AnswerExactMatch`` metric then reports how
    often the model's response matches either the original answer (parametric
    accuracy) or the substituted answer (context accuracy).

    Dataset: `pminervini/NQ-Swap <https://huggingface.co/datasets/pminervini/NQ-Swap>`_

    Required evaluation data keys:
        - ``id``: unique identifier
        - ``question``: factual question
        - ``sub_context``: substituted passage presented to the model
        - ``org_answer``: list of reference answers from the original context (parametric target)
        - ``sub_answer``: list of reference answers from the substituted context (context target)

    Optional evaluation data keys:
        - ``org_context``: original passage (not used during generation)
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]) -> None:
        """Validates that a single evaluation instance contains required fields.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance.

        Raises:
            ValueError: If ``id`` or any required key is missing, or if required
                fields are null/NaN.
        """
        if "id" not in evaluation_data:
            raise ValueError("The evaluation data must include an 'id' key.")

        missing_keys = [k for k in _EVALUATION_REQ_KEYS if k not in evaluation_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if any(
            evaluation_data[k] is None
            or (
                isinstance(evaluation_data[k], float) and math.isnan(evaluation_data[k])
            )
            for k in _EVALUATION_REQ_KEYS
        ):
            raise ValueError("Some required fields are null.")

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[tuple[str, str], str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Generates model responses for NQ-Swap instances.

        Each prompt presents the substituted context passage together with the
        question and asks for a short phrase answer. The model's response is
        left as free-form text for downstream evaluation with
        ``AnswerExactMatch``.

        Args:
            model_or_pipeline: HuggingFace model or ``SteeringPipeline`` instance.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters.
            runtime_overrides: Optional runtime parameter overrides for steering controls.
            **kwargs: Must include ``batch_size`` (int).

        Returns:
            List of generation dictionaries, each containing:

                - ``response``: Raw text response from the model.
                - ``question_id``: Instance identifier.
                - ``question``: Original question text.
                - ``org_answer``: List of parametric reference answers.
                - ``sub_answer``: List of context reference answers.
        """
        if not self.evaluation_data:
            return []

        gen_kwargs = dict(gen_kwargs or {})
        batch_size: int = int(kwargs["batch_size"])

        prompt_data = [
            {
                "id": instance["id"],
                "prompt": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            sub_context=instance["sub_context"],
                            question=instance["question"],
                        ),
                    }
                ],
            }
            for instance in self.evaluation_data
        ]

        responses = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            evaluation_data=self.evaluation_data,
            batch_size=batch_size,
        )

        generations = [
            {
                "response": response.split("\n")[0].strip() if response else response,
                "question_id": instance["id"],
                "question": instance["question"],
                "org_answer": list(instance["org_answer"]),
                "sub_answer": list(instance["sub_answer"]),
            }
            for instance, response in zip(self.evaluation_data, responses)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against parametric and context reference answers.

        Args:
            generations: List of generation dictionaries returned by ``generate()``.

        Returns:
            Dictionary of scores keyed by ``metric_name``. Each value is a dict
            with at least ``parametric_accuracy`` and ``context_accuracy`` keys.
        """
        eval_data = {
            "responses": [g["response"] for g in generations],
            "org_answers": [g["org_answer"] for g in generations],
            "sub_answers": [g["sub_answer"] for g in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)
        return scores

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports NQ-Swap evaluation results to structured JSON files.

        Creates two output files:

            1. ``responses.json``: Per-question model responses for each steering
               pipeline together with reference answers.
            2. ``scores.json``: Aggregate metric scores for each steering pipeline.

        Args:
            profiles: Dictionary containing evaluation results from all tested pipelines.
            save_dir: Directory path where results should be saved.
        """
        folder_path = Path(save_dir)
        folder_path.mkdir(parents=True, exist_ok=True)

        steering_methods = []
        predictions: dict[str, list[str]] = {}
        questions: list[str] | None = None
        org_answers: list[list[str]] | None = None
        sub_answers: list[list[str]] | None = None

        for method, runs in profiles.items():
            first_run = runs[0] if isinstance(runs, list) else runs
            gens = first_run["generations"]
            steering_methods.append(method)
            predictions[method] = [g["response"] for g in gens]

            if questions is None:
                questions = [g["question"] for g in gens]
                org_answers = [g["org_answer"] for g in gens]
                sub_answers = [g["sub_answer"] for g in gens]

        responses = [
            {
                "question": questions[i],
                "org_answer": org_answers[i],
                "sub_answer": sub_answers[i],
                **{method: predictions[method][i] for method in steering_methods},
            }
            for i in range(len(questions))
        ]

        with open(folder_path / "responses.json", "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)

        scores_only = {
            method: [
                {k: v for k, v in run.items() if k != "generations"}
                for run in (runs if isinstance(runs, list) else [runs])
            ]
            for method, runs in profiles.items()
        }

        with open(folder_path / "scores.json", "w", encoding="utf-8") as f:
            json.dump(scores_only, f, indent=4, ensure_ascii=False)
