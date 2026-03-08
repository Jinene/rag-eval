import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision
)

class Evaluator:

    def evaluate(self, dataset):

        results = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision
            ]
        )

        return results

    def save(self, results):

        df = results.to_pandas()
        df.to_csv("results.csv")
