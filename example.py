from phoenix.otel import register
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# configure the Phoenix tracer
tracer_provider = register(
  project_name="Demo App", # Default is 'default'
  auto_instrument=True, # See 'Trace all calls made to a library' below
)
tracer = tracer_provider.get_tracer(__name__)

client = openai.OpenAI()

examples = [
    "What is the weather today like?",
    "Did Rishabh Pant create history at Leeds?",
    "KL Rahul proved that he can bat in all the forms of cricket",
    "I went to the office today"
]

def classify(example):
    system_prompt = "You are a grammar expert. You need to classify the given user query into one of these two types: Statement or Question. Please output only the type and nothing else. I dont want any explanation or an"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a haiku."}],
    )
    answer = response.choices[0].message.content