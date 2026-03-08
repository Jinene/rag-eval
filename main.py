import json

from rag.retriever import Retriever
from rag.generator import generate
from evaluation.evaluator import Evaluator

with open("data/documents.txt") as f:
    documents = f.read().split("\n")

with open("data/dataset.json") as f:
    dataset = json.load(f)

retriever = Retriever(documents)

results = []

for item in dataset:

    question = item["question"]
    ground_truth = item["answer"]

    context = retriever.retrieve(question)

    answer = generate(question, "\n".join(context))

    results.append({
        "question": question,
        "answer": answer,
        "contexts": context,
        "ground_truth": ground_truth
    })

evaluator = Evaluator()

evaluation = evaluator.evaluate(results)

evaluator.save(evaluation)

print("Evaluation complete. Results saved.")
