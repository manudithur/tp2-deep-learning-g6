
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, ContextualRelevancyMetric, \
    ContextualPrecisionMetric

class Metrics:

    def __init__(self, prompt, actual_output, retrieved_docs, expected_output):
        self.test_case = LLMTestCase(
            input=prompt,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieved_docs
        )
        
    def evaluate_faithfulness(self):
        #FAITHFULNESS METRIC
        metric = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(self.test_case)
        print("Faithfulness Metric:\n")
        print(metric.score)
        print(metric.reason)

    def evaluate_answer_relevancy(self):
        #ANSWER RELEVANCY
        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(self.test_case)
        print("Answer Relevancy Metric:\n")
        print(metric.score)
        print(metric.reason)

    def evaluate_contextual_precision(self):
        #CONTEXTUAL PRECISION
        metric = ContextualPrecisionMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(self.test_case)
        print("Contextual Precision Metric:\n")
        print(metric.score)
        print(metric.reason)

    def evaluate_contextual_recall(self):
        #CONTEXTUAL RELEVANCY
        metric = ContextualRelevancyMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(self.test_case)
        print("Contextual Relevancy Metric:\n")
        print(metric.score)
        print(metric.reason)

