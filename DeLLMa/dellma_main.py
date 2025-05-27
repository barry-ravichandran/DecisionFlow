import json
import os
from tqdm import tqdm
import sys
from DeLLMa.utils.data_utils import get_combinations, FRUITS, STOCKS
from functools import partial
from collections import defaultdict
from DeLLMa.dellma_agent.farmagent import FarmAgent
from DeLLMa.dellma_agent.tradeagent import TradeAgent
from DeLLMa.dellma_agent.agent import (
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)
from DeLLMa.utils.prompt_utils import (
    inference,
    majority_voting_inference,
    chain_of_thought_inference,
    math_inference,
)

from DeLLMa.dellma_DecisionFlow.dellma_generate import DecisionFlow_reason

def agriculture_stocks_function(dataset_type, year_par, result_path_par, dellma_mode_par, model, model_path, temperature):
    result_path = result_path_par
    year = year_par
    dellma_mode = dellma_mode_par
    if dataset_type == "agriculture":
        products = FRUITS["2021"]
        domain = "agriculture"
        agent_init_fct = partial(
            FarmAgent,
            raw_context_fname=f"fruit-sept-2021.txt",
        )
        budget = 10

    elif dataset_type == "stocks":
        products = STOCKS
        domain = "stocks"
        agent_init_fct = partial(
            TradeAgent,
            history_length=24,
        )
        budget = 10000
        year = ""
    action_config = ActionConfig(budget=budget)
    result_folder = f"DeLLMa_results/{result_path}/{domain}/{year}/{dellma_mode}"
    state_enum_mode = "base"
    preference_config = PreferenceConfig()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if dataset_type == "agriculture":
        agent_name = "farmer"
    else:
        agent_name = "trader"
    combs = get_combinations(agent_name, source_year=year)
    pbar = tqdm(combs)
    for choices in pbar:
        pbar.set_description(f"Processing {choices}")
        agent = agent_init_fct(
            choices=choices,
            state_config=StateConfig(state_enum_mode),
            action_config=action_config,
            preference_config=preference_config,
        )
        if dellma_mode == "cot":
            prompts = agent.prepare_chain_of_thought_prompt()
        else:
            prompts, context, actions, state, preference = agent.prepare_dellma_prompt()
        if type(prompts) == str:
            prompts = [prompts]
        if dellma_mode == "cot":
            inference_fct = partial(
                chain_of_thought_inference,
                system_content=agent.system_content,
            )
        elif dellma_mode == "self-consistency":
            inference_fct = partial(
                majority_voting_inference,
                system_content=agent.system_content,
                num_samples=5,
            )
        elif dellma_mode == "decisionflow":
            inference_fct = partial(
                math_inference,
                system_content = agent.system_content,
            )
        else:
            inference_fct = partial(
                inference,
                system_content=agent.system_content,
            )

        path = f"{result_folder}/{'-'.join(choices)}"
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            continue
        if not os.path.exists(path + "/prompt"):
            os.makedirs(path + "/prompt")
        if not os.path.exists(path + "/response"):
            os.makedirs(path + "/response")

        if dellma_mode == "cot":
            output = inference_fct(
                chain=prompts, 
                temperature=temperature,
                model=model, 
                model_path=model_path
            )
            response = output["response"]
            prompt = output["query"]
            decision = response["decision"]
            with open(f"{path}/prompt/prompt.json", "w") as f:
                json.dump(prompt, f, indent=4)
            with open(f"{path}/response/response.json", "w") as f:
                json.dump(response, f, indent=4)

        elif dellma_mode == "decisionflow":
            # generate detailed_information
            detailed_infor_path = f"{path}/prompt/detailed_infor.json"
            if not os.path.exists(detailed_infor_path):
                try:
                    prompts, detailed_infor = inference_fct(
                        query=prompts,
                        temperature=temperature,
                        model=model,
                        model_path=model_path,
                        agent_name=agent_name,
                        context=context,
                        actions=actions,
                        state=state,
                        preference=preference,
                        is_generate_prompt=True,
                        path=path,
                    )
                except Exception as e:
                    print(e)
                    raise Exception("Prompt generation failed")
                with open(f"{path}/prompt/detailed_infor.json", "w") as f:
                    json.dump(detailed_infor, f, indent=4)
            with open(detailed_infor_path, "r") as f:
                detailed_infor = json.load(f)

            # load detailed information and generate prompt
            math_reason_class = DecisionFlow_reason(
                detailed_infor, agent_name,
            )
            prompt, detailed_infor = math_reason_class(
                actions,
                preference,
                temperature=temperature,
                model=model,
                model_path=model_path,
                path=path,
            )
            with open(f"{path}/prompt/prompt.txt", "w") as f:
                f.write(prompt)
            with open(f"{path}/prompt/detailed_assignment.json", "w") as f:
                json.dump(detailed_infor, f, indent=4)

            # respond to the prompt
            response, detailed_infor = inference_fct(
                query=prompt,
                temperature=temperature,
                model=model,
                model_path=model_path,
                is_generate_prompt=False,
            )
            with open(f"{path}/response/response.json", "w") as f:
                json.dump(response, f, indent=4)

        else:
            for i, prompt in enumerate(prompts):
                # save dellma prompt
                with open(f"{path}/prompt/prompt_{i}.txt", "w") as f:
                    f.write(prompt)
                response = inference_fct(
                    query=prompt,
                    temperature=temperature,
                    model=model,
                    model_path=model_path,
                )
                # save dellma response
                with open(f"{path}/response/response_{i}.json", "w") as f:
                    json.dump(response, f, indent=4)
