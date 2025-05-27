import os
import json
from typing import List, Dict, Callable
from time import sleep
import pandas as pd
import re

import openai
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from secret import OPENAI_API_KEY, OPENAI_API_BASE, CHATGPT_API

SUMMARY_PROMPT = (
    "You are a helpful agricultural expert studying a report published by the USDA"
)
ANALYST_PROMPT = "You are a helpful agricultural expert helping farmers decide what produce to plant next year."


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"


def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
    model: str = "open_source",
    model_path: str = "",
):
    success = False
    while not success:
        if model == "open_source":
            try:
                env = Environment(loader=FileSystemLoader("."))
                template = env.get_template("chat_template.jinja")
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query + "<json>"},
                ]
                prompt = template.render(messages=messages)
                openai_api_key = OPENAI_API_KEY
                openai_api_base = OPENAI_API_BASE
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )

                response = client.completions.create(
                    model=model_path,
                    prompt=prompt,
                    stream=False,
                    temperature=temperature,
                    max_tokens=4096,
                )
                success = True
            except Exception as e:
                print(e)
                sleep(10)
        elif model == "closed_source":
            try:
                CLIENT = OpenAI(
                    api_key=CHATGPT_API
                )
                response = CLIENT.chat.completions.create(
                    model=model_path,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query + "<json>"},
                    ],
                    temperature=temperature,
                    )
                success = True
            except Exception as e:
                print(e)
                sleep(10)

    if model == "open_source":
        try:
            response = response.choices[0].text
            # print(response)
        except Exception as e:
            print(e)
            response = ""
    elif model == "closed_source":
        try:
            response = response.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""
    try:
        response_ = json.loads(response.lower().strip())
    except:
        try:
            response_ = extractJSONToDict(response_)
        except:
            response_ = response

    return response_


def extractJSONToDict(response: str, language_identifer_optional=True):
    if not isinstance(response, str):
        print(f"Error: Expected a string, but got {type(response)}")
        raise TypeError("Response should be a string containing JSON data.")
    # Define the pattern
    if language_identifer_optional:
        pattern = r"```(?:json)?\s*(.*?)```"
    else:
        pattern = r"```json\s*(.*?)```"
    matches = re.findall(pattern, response, flags=re.DOTALL)

    json_str = None
    if matches:
        json_str = matches[0]
    else:
        print("No JSON found in the text.")
        print(json_str)
        raise Exception("No JSON Found")

    selection = None
    if json_str:
        try:
            selection = json.loads(json_str)
        except Exception as e:
            print(e)
            selection = ""
            raise Exception("Unable to instantiate JSON.")

    return selection


def math_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
    model: str = "closed_source",
    model_path: str = "",
    agent_name: str = "framer",
    context: str = "",
    actions: str = "",
    state: str = "",
    preference: str = "",
    is_generate_prompt: bool = True,
    path: str = "",
    detailed_infor: dict = {},
    sample_size: float = 16,
    minibatch_size: float = 32,
    overlap_pct: float = 0.25,
):
    if is_generate_prompt:
        from DeLLMa.dellma_DecisionFlow.dellma_generate import DecisionFlow_class
        math_class = DecisionFlow_class(
            prompt=query,
            temperature=temperature,
            agent_name=agent_name,
            context=context,
            actions=actions,
            state=state,
            preference=preference,
            path=path,
        )
        prompts, detailed_infor = math_class(model=model, model_path=model_path)
        return prompts, detailed_infor
    else:
        response = inference(
            query, system_content, temperature, model=model, model_path=model_path
        )
        return response, ""


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.7,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
    model: str = "llama",
    model_path: str = "",
):
    responses = []
    for _ in range(num_samples):
        if use_chain_of_thought:
            response = chain_of_thought_inference(
                chain=query, system_content=system_content, temperature=temperature
            )["response"]
        else:
            response = inference(query, system_content, temperature, model=model, model_path=model_path)
        responses.append(response)
    decisions = [r["decision"] for r in responses]
    majority_decision = max(set(decisions), key=decisions.count)
    response = {
        "decision": majority_decision,
        "explanation": responses,
    }
    return response


def chain_of_thought_inference(
    chain: List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.5,
    model: str = "llama",
    model_path: str = "",
):
    history = {}
    for query in chain:
        if isinstance(query, str):
            response = inference(query, system_content, temperature, model=model, model_path=model_path)
        else:
            previous_results = [history[k] for k in history.keys()]
            query = query(*previous_results)
            response = inference(query, system_content, temperature, model=model, model_path=model_path)
        history[query] = response

    return {
        "query": [{"prompt": q, "response": r} for q, r in history.items()],
        "response": response,
    }


def summarize(
    fname: str, products: List[str], temperature: float = 0.0
) -> Dict[str, str]:
    products = sorted(p.lower() for p in products)
    summary_fname = fname.split(".")[0] + "-" + "-".join(products) + ".json"
    if os.path.exists(summary_fname):
        # print(f"Summary file {summary_fname} already exists.")
        return json.load(open(summary_fname))

    report = open(fname).read()
    query = f"Below is an agriculture report published by the USDA:\n\n{report}\n\n"

    format_instruction = f"""Please write a detailed summary of the report.

You should format your response as a JSON object. The JSON object should contain the following keys:
    overview: a string that describes, in detail, the overview of the report. Your summary should focus on factors that affect the overall furuit and nut market.
    """
    for p in products:
        format_instruction += f"""
    {p}: a string that describes, in detail, information pertaining to {p} in the report. You should include information on {p} prices and production, as well as factors that affect them. 
        """
    query = format_query(query, format_instruction)
    response = inference(query, SUMMARY_PROMPT, temperature)
    try:
        response = json.loads(response.lower())
    except:
        response = response
    with open(summary_fname, "w") as f:
        json.dump(response, f, indent=4)
    return response
