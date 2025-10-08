import openai
client = openai.OpenAI()

def extract_xml_field(response: str, output_field: str) -> str:
    try: 
        return response.split(f"<{output_field}>")[1].split(f"</{output_field}>")[0]
    except:
        pass

def get_response_from_openai(query: str, model: str, XML_INSTRUCTIONS: str) -> str:
    # print("xml:", XML_INSTRUCTIONS(output_field))
    # print("query:", query)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": XML_INSTRUCTIONS},
            {"role": "user",   "content": query},
        ],
    )
    return resp.choices[0].message.content