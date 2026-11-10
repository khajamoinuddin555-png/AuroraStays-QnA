
import sys
from pathlib import Path

import pandas as pd

from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase

from evaluation_llm import GreatLearningLLM
from rag_pipeline import ask_question


BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / "data" / "evaluation"
RESULTS_DIR = BASE_DIR / "data" / "evaluation_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_dataset(dataset_name):

    if dataset_name == "evaluation":
        dataset_path = EVALUATION_DIR / "evaluation.csv"
    elif dataset_name == "test":
        dataset_path = EVALUATION_DIR / "test.xlsx"
    else:
        raise ValueError("Choose either 'evaluation' or 'test'.")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix == ".csv":
        dataframe = pd.read_csv(dataset_path)
    else:
        dataframe = pd.read_excel(dataset_path)

    required_columns = [
        "question",
        "expected_answer",
        "relevant_context"
    ]

    if not all(column in dataframe.columns for column in required_columns):
        raise ValueError("Dataset is missing required columns.")

    print("=" * 70)
    print(f"AURORASTAYS QnA - {dataset_name.upper()} DATASET EVALUATION")
    print("=" * 70)

    print(f"\nTotal questions: {len(dataframe)}")

    # Initialize the Great Learning evaluation model
    evaluation_model = GreatLearningLLM()

    # Initialize all four DeepEval metrics
    metrics = {
        "faithfulness": FaithfulnessMetric(
            model=evaluation_model,
            threshold=0.5,
            async_mode=False
        ),
        "contextual_recall": ContextualRecallMetric(
            model=evaluation_model,
            threshold=0.5,
            async_mode=False
        ),
        "contextual_precision": ContextualPrecisionMetric(
            model=evaluation_model,
            threshold=0.5,
            async_mode=False
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            model=evaluation_model,
            threshold=0.5,
            async_mode=False
        )
    }

    results = []

    for index, row in dataframe.iterrows():

        question = str(row["question"])
        expected_answer = str(row["expected_answer"])
        reference_context = str(row["relevant_context"])

        print("\n" + "-" * 70)
        print(f"Question {index + 1}/{len(dataframe)}")
        print("-" * 70)
        print("Question:", question)

        # Generate an answer using the RAG pipeline
        rag_result = ask_question(question)

        actual_answer = rag_result["answer"]

        # Extract the actual retrieved policy chunks
        retrieved_context = [
            chunk["text"]
            for chunk in rag_result["retrieved_chunks"]
        ]

        # Create a DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=[reference_context],
            retrieval_context=retrieved_context
        )

        print("\nGenerated Answer:")
        print(actual_answer)

        row_result = {
            "question": question,
            "actual_answer": actual_answer
        }

        # Run all four evaluation metrics
        for metric_name, metric in metrics.items():

            print(f"\nRunning {metric_name}...")

            metric.measure(test_case)

            score = metric.score
            reason = metric.reason

            row_result[f"{metric_name}_score"] = score
            row_result[f"{metric_name}_reason"] = reason

            print(f"{metric_name} score: {score}")

        results.append(row_result)

    # Save detailed evaluation results
    results_df = pd.DataFrame(results)

    output_file = RESULTS_DIR / f"{dataset_name}_results.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")

    # Calculate average scores
    metric_columns = [
        "faithfulness_score",
        "contextual_recall_score",
        "contextual_precision_score",
        "answer_relevancy_score"
    ]

    average_scores = results_df[metric_columns].mean()

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nDataset: {dataset_name}")
    print(f"Total questions evaluated: {len(results_df)}")

    for metric_name, score in average_scores.items():
        print(f"{metric_name}: {score:.4f}")

    print("\nResults saved to:")
    print(output_file)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python evaluate_rag.py evaluation")
        print("   or: python evaluate_rag.py test")
        sys.exit(1)

    evaluate_dataset(sys.argv[1].lower())
