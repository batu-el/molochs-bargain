import openai
client = openai.OpenAI()
import json
import os

XML_INSTRUCTIONS = lambda x: (
    "You are a helpful assistant. "
    "When you reply, first enclose your chain‑of‑thought inside "
    f"<think> … </think>, followed immediately by your final {x} inside "
    f"<{x}> … </{x}>. Do not output anything else."
)

def extract_xml_field(response: str, output_field: str) -> str:
    try: 
        return response.split(f"<{output_field}>")[1].split(f"</{output_field}>")[0]
    except:
        pass

def get_response_from_openai(query: str, model: str, output_field: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": XML_INSTRUCTIONS(output_field)},
            {"role": "user",   "content": query},
        ],
    )
    return resp.choices[0].message.content

def get_data(task_name, model_name, method_name):
    path =  f"res/{task_name}/{model_name}/{method_name}/test_step2.json"
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    data = [
        {
            "prompt": item["prompt"][0],
            "completion": item["completion"][0],
            "player_candidates": item["player_candidates"][0],
            "player_thinks": item["player_thinks"][0]
        }
        for item in data
    ]   
    return data