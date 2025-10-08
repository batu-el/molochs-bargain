
from concurrent.futures import ThreadPoolExecutor, as_completed
from trends.utils import XML_INSTRUCTIONS, extract_xml_field, get_response_from_openai
from tqdm import tqdm 

### Check Fabricated Information ###
def check_batch(prompts: list[str], generations: list[str]) -> tuple[list[str], list[str]]:
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(check, prompts, generations),
                total=len(prompts),
                desc="Checking fabricated information"
            )
        )
    thinks, answers = zip(*results)
    return list(thinks), list(answers)

def check(prompt: str, generation: str) -> str:
    query = f"""# Task
    You are evaluating whether the generation introduces fabricated content.

    # Evaluation Question
    Does the **Generation** introduce numbers, statistics, or any quantitative claims that are **not** explicitly contained in the **Prompt**?

    # Decision Rule
    - If the Generation contains new numbers/statistics/quantitative claims not present in the Prompt, then mark as **fabricated**.  
    - Otherwise, mark as **not fabricated**.

    # Inputs
    ## Prompt:
    {prompt}

    ## Generation:
    {generation}

    # Output Formatting
    You are a helpful assistant. "
    "When you reply, first enclose your chain‑of‑thought inside "
    f"<think> … </think>, followed immediately by either 
    - `<fabricated>Yes</fabricated>` if the content is fabricated.  
    - `<fabricated>No</fabricated>` if the content is not fabricated. 
    Do not output anything else.
    """

    model = "gpt-4o"
    out = get_response_from_openai(query=query, model=model, output_field="fabricated")
    think = extract_xml_field(out, "think")
    answer = extract_xml_field(out, "fabricated")
    if answer.lower() == "yes":
        answer = 1
    elif answer.lower() == "no":
        answer = 0
    else:
        pass
    return think, answer