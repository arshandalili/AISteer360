"""HaluEval QA use case for evaluating knowledge grounding vs. hallucination."""
import json
import math
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVALUATION_REQ_KEYS = [
    "knowledge",
    "question",
    "right_answer",
    "hallucinated_answer",
]

_PROMPT_TEMPLATE = (
    "Given the following knowledge, answer the question with a short phrase.\n\n"
    "Knowledge: {knowledge}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class HaluEvalQA(UseCase):
    """HaluEval QA knowledge grounding evaluation use case.

    Evaluates whether steering methods can guide a model toward grounded,
    knowledge-faithful answers or toward plausible-but-incorrect hallucinations.
    Each evaluation instance provides a background knowledge passage, a question,
    a correct answer extracted from that passage (``right_answer``), and a
    hallucinated alternative (``hallucinated_answer``).

    The model is always prompted with the knowledge passage. The
    ``HallucinationJudge`` metric then reports:

    - ``grounded_accuracy``: fraction of responses the judge classifies as matching the grounded answer.
    - ``hallucination_rate``: fraction of responses the judge classifies as matching the hallucinated answer.

    Dataset: `pminervini/HaluEval <https://huggingface.co/datasets/pminervini/HaluEval>`_
    (subset ``"qa"``, split ``"data"``).

    Required evaluation data keys:
        - ``id``: unique identifier
        - ``knowledge``: background passage presented to the model
        - ``question``: factual question
        - ``right_answer``: grounded reference answer (knowledge-faithful)
        - ``hallucinated_answer``: plausible but incorrect reference answer
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
            or (isinstance(evaluation_data[k], float) and math.isnan(evaluation_data[k]))
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
        """Generates model responses for HaluEval QA instances.

        Each prompt presents the knowledge passage and question and asks for a
        short phrase answer. Responses are left as free-form text for downstream
        evaluation with ``HallucinationJudge``.

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
                - ``right_answer``: Single-element list with the grounded reference answer.
                - ``hallucinated_answer``: Single-element list with the hallucinated reference answer.
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
                            knowledge=instance["knowledge"],
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
                "response": response,
                "question_id": instance["id"],
                "question": instance["question"],
                "right_answer": [instance["right_answer"]],
                "hallucinated_answer": [instance["hallucinated_answer"]],
            }
            for instance, response in zip(self.evaluation_data, responses)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses via the configured metrics.

        Args:
            generations: List of generation dictionaries returned by ``generate()``.

        Returns:
            Dictionary of scores keyed by ``metric_name``. Each value is a dict
            with at least ``grounded_accuracy`` and ``hallucination_rate`` keys
            (when using ``HallucinationJudge``).
        """
        eval_data = {
            "responses":            [g["response"]            for g in generations],
            "questions":            [g["question"]            for g in generations],
            "right_answers":        [g["right_answer"]        for g in generations],
            "hallucinated_answers": [g["hallucinated_answer"] for g in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)
        return scores

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports HaluEval QA evaluation results to structured JSON files.

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
        right_answers: list[str] | None = None
        hallucinated_answers: list[str] | None = None

        for method, runs in profiles.items():
            first_run = runs[0] if isinstance(runs, list) else runs
            gens = first_run["generations"]
            steering_methods.append(method)
            predictions[method] = [g["response"] for g in gens]

            if questions is None:
                questions = [g["question"] for g in gens]
                right_answers = [g["right_answer"][0] for g in gens]
                hallucinated_answers = [g["hallucinated_answer"][0] for g in gens]

        responses = [
            {
                "question": questions[i],
                "right_answer": right_answers[i],
                "hallucinated_answer": hallucinated_answers[i],
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
