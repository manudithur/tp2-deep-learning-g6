
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, ContextualRelevancyMetric, \
    ContextualPrecisionMetric


def evaluate_faithfulness(prompt, response, retrieved_docs):
    test_case = LLMTestCase(
        input=prompt,
        actual_output=response,
        retrieval_context=retrieved_docs
    )
    #FAITHFULNESS METRIC
    metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )
    metric.measure(test_case)
    print("Faithfulness Metric:\n")
    print(metric.score)
    print(metric.reason)

def evaluate_answer_relevancy(prompt, response, retrieved_docs):
    test_case = LLMTestCase(
        input=prompt,
        actual_output=response,
        retrieval_context=retrieved_docs
    )
    #ANSWER RELEVANCY
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )
    metric.measure(test_case)
    print("Answer Relevancy Metric:\n")
    print(metric.score)
    print(metric.reason)

def evaluate_contextual_precision(prompt, response, retrieved_docs, expected_output):
    test_case = LLMTestCase(
        input=prompt,
        actual_output=response,
        expected_output=expected_output,
        retrieval_context=retrieved_docs
    )
    #CONTEXTUAL PRECISION
    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )
    metric.measure(test_case)
    print("Contextual Precision Metric:\n")
    print(metric.score)
    print(metric.reason)

def evaluate_contextual_recall(prompt, response, retrieved_docs, expected_output):
    test_case = LLMTestCase(
        input=prompt,
        actual_output=response,
        expected_output=expected_output,
        retrieval_context=retrieved_docs
    )
    #CONTEXTUAL RELEVANCY
    metric = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )
    metric.measure(test_case)
    print("Contextual Relevancy Metric:\n")
    print(metric.score)
    print(metric.reason)

