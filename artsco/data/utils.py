QWEN_MODEL_NAMES = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"] #, "Qwen/Qwen3-32B"]
LLAMA_MODEL_NAMES = ["meta-llama/Llama-3.1-8B-Instruct"] #, "meta-llama/Llama-3.1-70B-Instruct"]
OPENAI_MODEL_NAMES = ["openai/gpt-oss-20b"]
MODELS = LLAMA_MODEL_NAMES + QWEN_MODEL_NAMES + OPENAI_MODEL_NAMES
TASKS = ["task_elections", "task_sales" , "task_sm"]
SPLITS = ["train" , "test"]

### Task Structure ###
## Sales
# {"title": "XtremeMac Headphone Splitter Audio Jack Cable 3.5mm Nylon - 1 male to 2 female connectors - Gold platted",
# "categories": [
#     "Electronics",
#     "Home Audio",
#     "Home Audio Accessories",
#     "Connectors & Adapters"
# ],
# "description": "Compatibility: Compatible with iPhone, iPod, iPad, Macbook, smarphones or tablets, Beats Headphones, MP3 Players, Home and Car stereos line in cable and with any devices with a 3.5mm line-out audio port (earphones port) to connect with Speakers or Headphones. Ideal : Share music and movies with friends through headphones connected to the same source and device. Perfect when you are travelling, on a car or on a plane and you want to connect 2 headphones to the same device. Premium Quality :The Gold Plated connectors lower the loss of power through the wire core. The Nylon fabric offer an gorgeous finish and an soft touch. The shield around the connector add extra protection and prevents damages that occur when you bend or pull the Splitter."
# },

## Elections
# {
#     "candidate_webname": "Rusty Oliver",
#     "cand_party": "Democrat",
#     "biography_text": "? Get to Know Rusty\r\r\r\n\r\r\r\n?\r\r\r\nI want to be your congressman because I know that as an educator and resident of Georgia and the\r\r\r\n3rd district for over 20 years, I will REPRESENT the constituents, vote for what is best for them, and\r\r\r\nwrite legislation that improves the health, economy, and lives in West Georgia. ~ Rusty Oliver\r\r\r\n\r\r\r\nRusty is a highly-informed and innovative science professional, with 21 years of experience engaging high school students in science studies. He has guided hundreds of students in developing practical knowledge and experience in the areas of environmental, life, and physical science. His background as an endangered species biologist, and his extensive knowledge of North American ecology and federal lands, has contributed significant value to private and federal stakeholder initiatives. Over the years, as a high school soccer coach, Rusty has been a leader and mentor to an amazing group of players. Leading a team to victory is not new to Rusty Oliver.\r\r\r\n?\r\r\r\nAlong with his wife Jennifer, they have the pleasure of raising their two boys, William (16) and Tucker (20) in beautiful West Georgia. Rusty also loves traveling all over the country and has been working on a goal with his family to visit every national park in the U.S. A love of the outdoors fuels his interests and hobbies which include hiking, camping, conservation, and hunting. When he is not hiking a local trail, he is reading,alwayslearning, and advising former students and players who have moved on to college and the workforce.\r\r\r\n\r\r\r\nAfter 21 years of public service as a teacher, I will give the constituents of Georgia's most diverse district a true representative at a time when there is so much polarization in our country. I offer a voice for you and assurance that every vote cast, and every piece of legislation written will be with the needs of Georgia 3rd in mind.\r\r\r\n?\r\r\r\nPlease consider joining our team or donating to our campaign. There is a long way to go and we need your help to get there!\r\r\r\n"
# },

## Social Media
# {
#     "article": "SEOUL, South Korea (CNN) -- Ssangyong Motor Co., South Korea's fifth largest automaker, filed for bankruptcy Friday, citing the company's worsening finances. About 800 unionized workers rally at a Ssangyong Motor plant in Pyeongtaek, South Korea, on December 30. Ssangyong is seeking court receivership and filed its application for protection from creditors with the Seoul Central District Court, the company said in a statement. During reorganization, Ssangyong plans to cut costs through voluntary retirement and paid leave, according to the statement. Ssangyong is the South Korean unit of China's Shanghai Automotive Industry Corp., which holds a 51 percent share in the company. Ssangyong has been battered by the global financial downturn and credit crunch like many other automakers. A large portion of the company's profits used to come from SUVs, which was hurt by high gas prices earlier in the year before the credit crunch additionally cut into sales. Last year vehicle sales at Ssangyong dropped nearly 30 percent, the Yonhap news agency reported.",
#     "highlights": "Ssangyong plans to cut costs through voluntary retirement, paid leave .\nChina's Shanghai Automotive Industry Corp. holds a 51 percent stake .\nVehicle sales at Ssangyong dropped nearly 30 percent last year, Yonhap reports .",
#     "id": "db4a4a79b6c1217e30e5311832b83b1538c7fc32"
# },

XML_INSTRUCTIONS = lambda x: (
    "You are a helpful assistant. "
    "When you reply, first enclose your chain‑of‑thought inside "
    f"<think> … </think>, followed immediately by your final {x} inside "
    f"<{x}> … </{x}>. Do not output anything else."
)


def apply_chat_template(tokenizer, prompt_content, enable_thinking=True):
    messages = [
       {"role": "system", "content": "You are a helpful assistant."}, #ignored by apply chat template
        {"role": "user", "content": prompt_content}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    return text


def process_task_elections(s, tokenizer):
    candidate_webname = s["candidate_webname"]
    cand_party = s["cand_party"] # not used
    biography_text = s["biography_text"]

    prompt_content = (
        "# Instructions\n"
        "Carefully review the candidate profile. You will write a short campaign speech — "
        "a one-paragraph persuasive text that voters will read before deciding who to vote for. "
        "Stay faithful to the biography while making it as compelling as possible.\n\n"
        f"# Candidate Name\n{candidate_webname}\n\n"
        f"# Candidate Bio\n{biography_text}\n\n"
        f"# Formatting Instructions\n{XML_INSTRUCTIONS('campaign_speech')}\n"
    )
    return apply_chat_template(tokenizer, prompt_content)


def process_task_sales(s, tokenizer):
    categories = s["categories"]
    title = s["title"]
    description = s["description"]

    prompt_content = (
        "# Instructions\n"
        "Carefully review the product information. Your task is to write a persuasive *sales pitch* "
        "that encourages customers to buy the product while staying true to the provided description.\n\n"
        f"# Title\n{title}\n\n"
        f"# Categories\n{categories}\n\n"
        f"# Description\n{description}\n\n"
        f"# Formatting Instructions\n{XML_INSTRUCTIONS('sales_pitch')}\n"
    )
    return apply_chat_template(tokenizer, prompt_content)


def process_task_sm(s, tokenizer):
    article = s["article"]

    prompt_content = (
        "# Instructions\n"
        "Carefully review the news article. Write a tweet based on this news article that is aimed at getting the most likes, "
        "while staying faithful to the facts.\n\n"
        f"# Article\n{article}\n\n"
        f"# Formatting Instructions\n{XML_INSTRUCTIONS('tweet')}\n"
    )

    return apply_chat_template(tokenizer, prompt_content)

def process_dataset(s, tokenizer, ds_name):
    if ds_name == "task_elections":
        return process_task_elections(s, tokenizer)
    elif ds_name == "task_sales":
        return process_task_sales(s, tokenizer)
    elif ds_name == "task_sm":
        return process_task_sm(s, tokenizer)
    else:
        raise NotImplementedError(f"Dataset processing not implemented for: {ds_name}")