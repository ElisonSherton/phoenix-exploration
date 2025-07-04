from phoenix.experiments import run_experiment
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
from typing import List, Literal
import phoenix as px
import yaml, os, json
from dotenv import load_dotenv
import openai
from textwrap import dedent
from pydantic import BaseModel, Field
from collections import defaultdict

load_dotenv()

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
DEBUG = config["debug"]

client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

phoenix_client = px.Client()
dataset_config = config["two_stage_dataset"]

df = phoenix_client.get_dataset(name=dataset_config["dataset_name"])

LLM_MODEL = os.environ["MODEL"]

# Create the first stage and second stage classification outputs
stage_one_category_pool = tuple(
    set([x.metadata[dataset_config["metadata_field"]] for x in df.examples.values()])
)
available_categories = "\n".join(stage_one_category_pool)

parent_child_map = defaultdict(lambda: [])
for example in df.examples.values():
    parent = example.metadata[dataset_config["metadata_field"]]
    child = example.output[dataset_config["output_field"]]
    if child not in parent_child_map[parent]:
        parent_child_map[parent].append(child)


stage_one_system_prompt = dedent(f"""
You are an expert in biology. You can distinguish and identify what is being talked about in a given piece of text.
You have the following list of categories at your perusal                       

{available_categories}

You will be provided with a user utterance. Your task is to map the user utterance to the closest category from the ones mentioned above. For example

USER_UTTERANCE: Lion is the most majestic of all
animal
""")

stage_two_system_prompt = dedent("""
You are an expert in biology. You can distinguish and identify what is being talked about in a given piece of text.
You have the following list of categories at your perusal                       

{available_categories}

You will be provided with a user utterance. Your task is to map the user utterance to the closest category from the ones mentioned above. For example

USER_UTTERANCE: Lion is the most majestic of all
lion
""")


class Response(BaseModel):
    explanation: str = Field(
        ...,
        description="A brief explanation for which category do you think maps to the given utterance.",
    )
    category: str = Field(
        ...,
        description="The category that you think the given utterance maps to.",
    )


def get_child(utterance, parent_category):
    try:
        response = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": stage_two_system_prompt.format(
                        available_categories = parent_child_map[parent_category]
                    ),
                },
                {
                    "role": "user",
                    "content": f"Provide the closest matching category for the following utterance.\nUTTERANCE: {utterance}",
                },
            ],
            response_format=Response,
        )
        response = Response(**json.loads(response.choices[0].message.content))
        explanation = response.explanation
        prediction = response.category
        return "success", explanation, prediction
    except Exception as e:
        print(str(e))
        return "error", "", str(e)


def get_category(utterance):
    try:
        response = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": stage_one_system_prompt},
                {
                    "role": "user",
                    "content": f"Provide the closest matching intent/intents for the following utterance.\nUTTERANCE: {utterance}",
                },
            ],
            response_format=Response,
        )
        response = Response(**json.loads(response.choices[0].message.content))
        parent_explanation = response.explanation
        parent_prediction = response.category

        error, child_explanation, child_category = get_child(
            utterance, parent_prediction
        )

        if error == "error":
            raise Exception(f"Problem in child predicion\n{child_category}")

        output = {
            "parent_category": parent_prediction,
            "parent_explanation": parent_explanation,
            "child_category": child_category,
            "child_explanation": child_explanation,
        }

        return "success", output, child_category
    except Exception as e:
        return "error", {}, str(e)


def task(input):
    error_message = ""
    error, explanation, prediction = get_category(input[dataset_config["input_field"]])
    if error == "error":
        error_message = prediction

    return {
        "prediction": prediction,
        "explanation": explanation,
        "error": error_message,
    }


def no_error(output) -> bool:
    return not bool(output.get("error"))


def accuracy(output, expected) -> bool:
    if DEBUG:
        print(f"Output: \n{output}\nExpected:\n{expected}")
    return output.get("prediction") == expected.get(dataset_config["output_field"])


tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

experiment = run_experiment(
    df,
    task=task,
    evaluators=[no_error, accuracy],
    dry_run=DEBUG,
    experiment_name="Structured-Output-2",
    experiment_description="Fixed the error in child prompt; Was providing duplicate categories. Now not doing that.",
)
