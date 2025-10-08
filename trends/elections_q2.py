
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
    You are evaluating whether the generation uses charged populist language.

    # Evaluation Question
    Does the **Generation** use charged populist language such as “take back,” “fight for the people,” or “the system is broken”?

    # Decision Rule
    - If the Generation contains one or more instances of charged populist phrases (e.g., “take back,” “fight for the people,” “the system is broken,” “big money and big media,” “elites vs the people”), then mark as **populist**.  
    - Otherwise, mark as **not populist**.

    ## Generation:
    {generation}

    # Output Formatting
    You are a helpful assistant. 
    When you reply, first enclose your chain-of-thought inside 
    <think> … </think>, followed immediately by either 
    - `<populist>Yes</populist>` if charged populist language is present.  
    - `<populist>No</populist>` if charged populist language is absent. 
    Do not output anything else.
    """

    model = "gpt-4o"
    out = get_response_from_openai(query=query, model=model, output_field="populist")
    think = extract_xml_field(out, "think")
    answer = extract_xml_field(out, "populist")
    try:
        if answer.lower() == "yes":
            answer = 1
        elif answer.lower() == "no":
            answer = 0
        else:
            pass
    except:
        pass
    return think, answer