"""Evaluate the legal RAG pipeline with DeepEval and an OpenRouter judge."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import database
import llm_connection
from rate_limits import DailyTokenLimiter, SessionTokenLimiter, estimate_eval_tokens
from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase
from rag_eval_goldens import GOLDENS

MetricFactory = Callable[[OpenRouterModel, float], object]

METRIC_REGISTRY: dict[str, MetricFactory] = {
    "contextual_recall": lambda model, threshold: ContextualRecallMetric(
        model=model, threshold=threshold
    ),
    "faithfulness": lambda model, threshold: FaithfulnessMetric(
        model=model, threshold=threshold
    ),
    "answer_relevancy": lambda model, threshold: AnswerRelevancyMetric(
        model=model, threshold=threshold
    ),
}

METRIC_LABELS = {
    "contextual_recall": "Contextual Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}


@dataclass
class EvalConfig:
    """Runtime settings for a RAG evaluation run."""

    limit: int | None = None
    offset: int = 0
    threshold: float = 0.5
    metrics: Sequence[str] = field(
        default_factory=lambda: tuple(METRIC_REGISTRY.keys())
    )
    print_results: bool = True
    judge_model: str = "tencent/hy3:free"
    identifier: str | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 when provided.")
        if self.offset < 0:
            raise ValueError("offset must be zero or greater.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if not self.metrics:
            raise ValueError("Select at least one metric.")

        unknown = [name for name in self.metrics if name not in METRIC_REGISTRY]
        if unknown:
            options = ", ".join(METRIC_REGISTRY)
            raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Choose from: {options}.")

        self.metrics = tuple(dict.fromkeys(self.metrics))


class OpenRouterJudge:
    """Provides one OpenRouter model instance for all DeepEval metrics."""

    def __init__(self, model: str = "tencent/hy3:free", api_key: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER") or os.environ.get(
            "OPENROUTER_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER in .env or OPENROUTER_API_KEY in the environment."
            )

        self.model_name = model
        self._model = OpenRouterModel(
            model=model,
            api_key=api_key,
            temperature=0.0,
            generation_kwargs={"extra_body": {"reasoning": {"enabled": True}}},
        )

    @property
    def model(self) -> OpenRouterModel:
        return self._model


class RAGPipeline:
    """Runs retrieval and generation for a single legal question."""

    def __init__(self, store: database.LegalVectorStore | None = None):
        self.store = store or database.get_store()

    def retrieve_context(self, query: str) -> list[str]:
        return [chunk["document"] for chunk in self.store.retrieve(query)]

    @staticmethod
    def _build_prompt(query: str, context: Sequence[str]) -> str:
        context_text = "\n\n".join(context)
        return f"""You are a helpful legal assistant. Use the context below to answer the question.

Context:
{context_text}

Question: {query}
Answer:"""

    @staticmethod
    def generate(query: str, context: Sequence[str]) -> str:
        return llm_connection.query_model_complete(
            RAGPipeline._build_prompt(query, context)
        )

    def run(self, query: str) -> tuple[str, list[str]]:
        retrieval_context = self.retrieve_context(query)
        return self.generate(query, retrieval_context), retrieval_context


class RAGEvaluator:
    """Builds test cases from goldens and evaluates the RAG response quality."""

    def __init__(
        self,
        config: EvalConfig,
        judge: OpenRouterJudge | None = None,
        pipeline: RAGPipeline | None = None,
        goldens: Sequence[Golden] = GOLDENS,
    ):
        self.config = config
        self.judge = judge or OpenRouterJudge(model=config.judge_model)
        self.pipeline = pipeline or RAGPipeline()
        self.goldens = goldens

    def _build_metrics(self) -> list:
        return [
            METRIC_REGISTRY[name](self.judge.model, self.config.threshold)
            for name in self.config.metrics
        ]

    def select_goldens(self) -> list[Golden]:
        selected = list(self.goldens[self.config.offset :])
        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        if not selected:
            raise ValueError("No goldens selected. Adjust offset or limit.")
        return selected

    def build_test_cases(self) -> list[LLMTestCase]:
        test_cases = []

        for golden in self.select_goldens():
            actual_output, retrieval_context = self.pipeline.run(golden.input)
            test_cases.append(
                LLMTestCase(
                    input=golden.input,
                    actual_output=actual_output,
                    expected_output=golden.expected_output,
                    retrieval_context=retrieval_context,
                )
            )

        return test_cases

    def run(self, session_state=None):
        test_cases = self.build_test_cases()
        result = evaluate(
            test_cases=test_cases,
            metrics=self._build_metrics(),
            identifier=self.config.identifier,
            display_config=DisplayConfig(print_results=self.config.print_results),
        )
        tokens = estimate_eval_tokens(test_cases)
        DailyTokenLimiter().record(tokens)
        if session_state is not None:
            SessionTokenLimiter().record(session_state, tokens)
        return result


def run_evaluation(config: EvalConfig, session_state=None):
    """Run an evaluation with the supplied configuration."""
    return RAGEvaluator(config=config).run(session_state=session_state)


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate the legal RAG pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N goldens after --offset.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N goldens before applying --limit.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score required for a metric to pass.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_REGISTRY.keys()),
        default=tuple(METRIC_REGISTRY.keys()),
        metavar="METRIC",
        help=(
            "Metrics to run. Choices: "
            + ", ".join(METRIC_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="tencent/hy3:free",
        help="OpenRouter model used as the LLM judge.",
    )
    parser.add_argument(
        "--identifier",
        default=None,
        help="Optional label for this test run in DeepEval output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress DeepEval console output.",
    )
    args = parser.parse_args()

    return EvalConfig(
        limit=args.limit,
        offset=args.offset,
        threshold=args.threshold,
        metrics=args.metrics,
        print_results=not args.quiet,
        judge_model=args.judge_model,
        identifier=args.identifier,
    )


def main() -> None:
    run_evaluation(_parse_args())


if __name__ == "__main__":
    main()
