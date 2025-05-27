from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import re
import os
import json
from typing import Dict
from collections import defaultdict

from DeLLMa.dellma_DecisionFlow.dellma_prompt import (
    agriculture_extraction,
    stock_extraction,
    math_attribute,
    math_attribute_error,
    agriculture_distribute,
    stocks_distribute,
    math_mean,
)

from secret import OPENAI_API_KEY, OPENAI_API_BASE, CHATGPT_API

belief2score: Dict[str, float] = {
    "very likely": 6,
    "likely": 5,
    "somewhat likely": 4,
    "somewhat unlikely": 3,
    "unlikely": 2,
    "very unlikely": 1,
}

def generate_output(temperature, model, model_path, prompt, system_message=""):
    if system_message == "":
        system_message = "You are an expert in mathematical modeling and optimization for agricultural decision-making."
    if model == "open_source":
        try:
            env = Environment(loader=FileSystemLoader("."))
            template = env.get_template("chat_template.jinja")
            messages = [
                {"role": "assistant", "content": system_message},
                {"role": "user", "content": prompt},
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

            # print(response.choices[0].text)
        except Exception as e:
            print(f"An error occurred: {e}")
        return response.choices[0].text
    elif model == "closed_source":
        client = OpenAI(
            api_key=CHATGPT_API
        )
        response = client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant skilled in generating accurate statements.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        statement = response.choices[0].message.content
        return statement

class DecisionFlow_class:

    def __init__(
        self,
        prompt,
        temperature,
        agent_name,
        context,
        actions,
        state,
        path,
        preference
    ):
        self.prompt = prompt
        self.temperature = temperature
        self.agent_name = agent_name
        self.context = context
        self.actions = actions
        self.state = state
        self.preference = preference
        prompt_file = f"{path}/prompt/detailed_infor.json"
        if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                _f = json.load(f)
            self.information = _f["Express"]
            self.action = [
                "Variables",
                "Express",
            ]
        else:
            self.information = {}

            self.action = [
                "Variables",
                "Extraction",
                "Attribute",
                "Objective",
                "Express",
            ]

    def extract_subjects(self, text):
        pattern = r"Action \d+\. (\w+):"
        subjects = re.findall(pattern, text)
        return subjects

    def __call__(self, model, model_path):
        # try:
        for action in self.action:
            retries = 3
            if action == "Variables":
                self.variables = self.extract_subjects(self.actions)
            elif action == "Extraction":
                numAttempts = 0
                self.extraction = []
                if self.agent_name == "farmer":
                    prompt = agriculture_extraction.format(
                        text=self.context,
                        variable=self.variables,
                    )
                    response = generate_output(
                        temperature=self.temperature,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    try:
                        self.extraction = extractJSONToDict(response)
                    except:
                        self.extraction = response
                    if self.extraction == "":
                        numAttempts += 1
                        raise ValueError("Empty extraction result")
                else:
                    prompt = stock_extraction.format(
                        text=self.context,
                    )
                    response = generate_output(
                        temperature=0,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    self.extraction = response
                if self.agent_name == "farmer":
                    self.information["factor"] = []
                    try:
                        for fruit_type in self.extraction["fruits"]:
                            fruit_name = fruit_type["Name"]
                            fruit_price = fruit_type["Price"]
                            fruit_yield = fruit_type["Yield"]
                            for factor in fruit_type["Influencing Factors"]:
                                self.information["factor"].append(
                                    {
                                        "Name": fruit_name,
                                        "Price": fruit_price,
                                        "Yield": fruit_yield,
                                        "Factor": factor,
                                    }
                                )
                    except:
                        self.information["factor"] = self.extraction
                else:
                    self.information["summary"] = self.extraction

            elif action == "Attribute":
                if self.agent_name == "trader":
                    continue
                numAttempts = 0
                try:
                    for num, single_extraction in enumerate(self.information["factor"]):
                        single_attribute = None
                        try:
                            prompt = math_attribute.format(
                                name=single_extraction["Name"],
                                factor=single_extraction["Factor"],
                            )
                            response = generate_output(
                                prompt=prompt,
                                model=model,
                                model_path=model_path,
                                temperature=self.temperature,
                            )
                            try:
                                single_attribute = extractJSONToDict(response)
                                self.information["factor"][num]["Attribute"] = (
                                    single_attribute["Attribute"]
                                )
                            except:
                                raise Exception("Attribute Error")
                        except Exception as e:
                            raise Exception("Attribute Error")
                except:
                    prompt = math_attribute_error.format(
                        information=self.information["factor"]
                    )
                    response = generate_output(
                        prompt=prompt,
                        model=model,
                        model_path=model_path,
                        temperature=self.temperature,
                    )
                    self.information = response
            elif action == "Objective":
                # print("**Objective**")
                if self.agent_name == "farmer":
                    self.objective = "I'm a farmer in California planning what fruit to plant next year. I would like to maximize my profit with '10' acres of land."
                else:
                    self.objective = "I'm a trader planning my next move. I would like to maximize my profit with '10' dollars."
        return self.prompt, {
            "Variables": self.variables,
            "Express": self.information,
        }

class DecisionFlow_reason:
    def __init__(self, detailed_infor, agent_name):
        self.detailed_infor = detailed_infor
        self.agent_name = agent_name
        self.action = [
            "Distribute", 
            "Mean", 
            "Profit", 
            "Choose"
            ]

    def __call__(
        self, actions, preference, temperature, model, model_path, path
    ):
        if self.agent_name == "farmer":
            response_path = f"{path}/response/response.json"

            if os.path.exists(response_path):
                with open(f"{path}/prompt/detailed_assignment.json", "r") as f:
                    information = json.load(f)
                self.distribute = information["Distribute"]
                self.profit = information["Profit"]
                prompt = """Please use the given information to solve the following problem:

            I'm a farmer in California planning what fruit to plant next year. I would like to maximize my profit with '10' acres of land.

            {actions}

            The profit for each fruit is here:
            {profit}

            {preference}
            """
                prompt = prompt.format(
                                actions=actions, profit=self.profit, preference=preference
                            )
                return prompt, {
                                "Distribute": self.distribute,
                                "Profit": self.profit,
                            }

            try:
                output = defaultdict(lambda: {"Price": "", "Yield": "", "Factors": []})
                for item in self.detailed_infor["Express"]["factor"]:
                    name = item["Name"]
                    price = item["Price"]
                    yield_ = item["Yield"]
                    factor_text = item["Factor"]
                    attribute = item["Attribute"]

                    output[name]["Price"] = price
                    output[name]["Yield"] = yield_

                    output[name]["Factors"].append(
                        {
                            "Factor": factor_text,
                            "Attribute": attribute,
                        }
                    )
                information = ""
                for fruit, info in output.items():
                    information += f"### {fruit.capitalize()}\n"
                    information += f"Price: {info['Price']}\nYield: {info['Yield']}\n"
                    information += "Impact factors:\n"
                    for factor in info["Factors"]:
                        information += f"- {factor['Factor']}"
                        information += f"  → Impact aspect: {factor['Attribute']}\n"
                    information += "\n\n"
            except:
                information = self.detailed_infor
            for action in self.action:
                if action == "Distribute":
                    prompt = agriculture_distribute
                    prompt = prompt.format(information=information)
                    response = generate_output(
                        temperature=temperature,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    try:
                        self.distribute = extractJSONToDict(response)
                    except Exception as e:
                        self.distribute = response

                elif action == "Mean":
                    prompt = math_mean
                    prompt = prompt.format(
                        information=information, distribute=self.distribute
                    )
                    response = generate_output(
                        temperature=temperature,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    try:
                        self.profit = extractJSONToDict(response)
                    except Exception as e:
                        self.profit = response

                elif action == "Choose":
                    # print("**Choose**")
                    prompt = """Please use the given information to solve the following problem:

I'm a farmer in California planning what fruit to plant next year. I would like to maximize my profit with '10' acres of land.

{actions}

The profit for each fruit is here: 
{profit}

{preference}
"""
                    prompt = prompt.format(
                        actions=actions, profit=self.profit, preference=preference
                    )
            return str(prompt), {
                "Distribute": self.distribute,
                "Profit": self.profit,
            }
        elif self.agent_name == "trader":
            for action in self.action:
                if action == "Distribute":
                    # print("**Distribute**")
                    prompt = stocks_distribute
                    self.distribute = {}
                    prompt = prompt.format(
                        information=self.detailed_infor
                    )
                    response = generate_output(
                        temperature=temperature,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    try:
                        self.distribute = extractJSONToDict(response)
                    except Exception as e:
                        self.distribute = response

                elif action == "Mean":
                    prompt = """
{information}
{distribute}
Calculate the increase of each stock:
increase = price / current_price

Then output the result like this:
```json
{{
  "AMD": 1.07,
  "DIS": 0.91
}}
```
You should give your answer in JSON format.
"""
                    prompt = prompt.format(
                        information=self.detailed_infor, distribute=self.distribute
                    )
                    response = generate_output(
                        temperature=temperature,
                        model=model,
                        model_path=model_path,
                        prompt=prompt,
                    )
                    try:
                        self.profit = extractJSONToDict(response)
                    except Exception as e:
                        self.profit = response

                elif action == "Choose":
                    # print("**Choose**")
                    prompt = """Please use the given information to solve the following problem:

I'm a trader planning my next move. I would like to maximize my profit.

{actions}

The profit for each stock is here: 
{profit}

{preference}
"""
                    prompt = prompt.format(
                        actions=actions, profit=self.profit, preference=preference
                    )
            return str(prompt), {
                "Distribute": self.distribute,
                "Profit": self.profit,
            }


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


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"
