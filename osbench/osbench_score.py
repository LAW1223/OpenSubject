from .prompt_generator import PromptGenerator
from .openai_util import ask_gpt4o
from .json_util import mllm_output_to_dict
import random
import json
import time

class OSBenchScore:
    def __init__(self, openai_url: str, openai_key: str) -> None:
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.prompt_generator = PromptGenerator()

    def evaluate(self, input_image_paths, instruction, with_manipulation=False, with_single=False):
        results_dict = {}

        max_tries = 10
        PF_scores = None
        SC_scores = None
        for try_idx in range(max_tries):
            try:
                PF_prompt = self.prompt_generator(instruction, task_type="opensubject_prompt_following", with_manipulation=with_manipulation, with_single=with_single)
                SC_prompt = self.prompt_generator(instruction, task_type="opensubject_subject_consistency", with_manipulation=with_manipulation, with_single=with_single)

                if with_manipulation:
                    PF_results = ask_gpt4o(input_image_paths[-2:], PF_prompt, self.openai_url, self.openai_key)

                else:
                    PF_results = ask_gpt4o(input_image_paths, PF_prompt, self.openai_url, self.openai_key)
                
                
                SC_results = ask_gpt4o(input_image_paths[-2:], SC_prompt, self.openai_url, self.openai_key)

                PF_scores = mllm_output_to_dict(PF_results)
                SC_scores = mllm_output_to_dict(SC_results)

                if PF_scores == "rate_limit_exceeded" or SC_scores == "rate_limit_exceeded":
                    raise Exception("rate_limit_exceeded")
                else:
                    break
            except Exception as e:
                backoff_time = 1  # Exponential backoff: 1, 2, 4 seconds
                # print(e)
                print(f"{e}, Attempt {try_idx+1} failed, retrying after {backoff_time} seconds...")

                # print(f"{e}, {instruction=}, Attempt {try_idx+1} failed, retrying after {backoff_time} seconds...")
                time.sleep(backoff_time)

        if PF_scores is None:
            guessed_value = random.randint(0, 10)
            print(f"Failed to find the json content in the string for {instruction}. Guess a value : {guessed_value=}.", flush=True)
            PF_scores = {'score': guessed_value, "reasoning": f"guess_if_cannot_parse | {PF_results}"}
        
        if SC_scores is None:
            guessed_value = random.randint(0, 10)
            print(f"Failed to find the json content in the string for {instruction}. Guess a value : {guessed_value=}.", flush=True)
            SC_scores = {'score': guessed_value, "reasoning": f"guess_if_cannot_parse | {SC_results}"}

        results_dict["PF_scores"] = PF_scores
        results_dict["SC_scores"] = SC_scores
        return results_dict