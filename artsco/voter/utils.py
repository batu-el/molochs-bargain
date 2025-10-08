import openai
client = openai.OpenAI()

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
    # print("xml:", XML_INSTRUCTIONS(output_field))
    # print("query:", query)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": XML_INSTRUCTIONS(output_field)},
            {"role": "user",   "content": query},
        ],
    )
    return resp.choices[0].message.content

def get_vote(query: str, model: str) -> str:
    return get_response_from_openai(query=query, model=model, output_field="vote")


import json

def load_persona20() -> list[str]:
    file_path: str = "artsco/data/persona/split1.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_persona100() -> list[str]:
    file_path: str = "artsco/data/persona/split1.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)