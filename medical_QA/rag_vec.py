import os, json
from time import time
from openai import OpenAI
import ingest_vec
from qdrant_client import models
from dotenv import load_dotenv
load_dotenv()

client_hpg = OpenAI(
    api_key=os.environ['HPG_API_KEY'],
    base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

client_q, model_handle = ingest_vec.load_index()

def search_vec(query, collection_name = "medicalQA-rag1020"):

    results = client_q.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        limit=10, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results


def build_prompt(query, search_results):
    prompt_template = """
    You're an assistant at the front desk of meadical information. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    entry_template = """
    answer_in_db: {text}
    cource_in_db: {source}
    focus_area_indb: {focus_area}
    """.strip()

    retrieved_context = ""

    for doc in search_results.points:
        # "**" unpacks the key-value pairs from the doc dictionary and passes them as keyword arguments to the format() method.
        # note to "**": the keys in the dictionary MUST match the placeholders in your template string
        retrieved_context += entry_template.format(**doc.payload) + "\n\n"

    prompt = prompt_template.format(question=query, context=retrieved_context).strip()
    return prompt


def llm(prompt, model = "gpt-oss-120b"):
    response = client_hpg.chat.completions.create(
        model=model, # model to send to the proxy
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats



eval_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = eval_prompt_template.format(question=question, answer_llm=answer)
    evaluation, tokens = llm(prompt, model = "gpt-oss-120b")

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_openai_cost(tokens):
    openai_cost = 0

    openai_cost = (
        tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
    ) / 1000

    return openai_cost


def rag(query, model = "gpt-oss-120b"):
    t0 = time()
    
    # retrieve information relevant to the query
    search_results = search_vec(query)

    # properly mix the query and retrieved info as a new prompt
    prompt = build_prompt(query, search_results)

    # send the new prompt to LLM
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(token_stats)
    openai_cost_eval = calculate_openai_cost(rel_token_stats)

    openai_cost = f"${round(openai_cost_rag + openai_cost_eval, 5)}" 

    answer_data = {
       "answer": answer,
       "model_used": model,
       "response_time": took,
       "relevance": relevance.get("Relevance", "UNKNOWN"),
       "relevance_explanation": relevance.get(
           "Explanation", "Failed to parse evaluation"
       ),
       "prompt_tokens": token_stats["prompt_tokens"],
       "completion_tokens": token_stats["completion_tokens"],
       "total_tokens": token_stats["total_tokens"],
       "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
       "eval_completion_tokens": rel_token_stats["completion_tokens"],
       "eval_total_tokens": rel_token_stats["total_tokens"],
       "openai_cost": openai_cost,
    }

    print(answer_data)
    
    return answer_data
