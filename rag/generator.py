from openai import OpenAI

client = OpenAI()

def generate(question, context):

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content
