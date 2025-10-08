from artsco.voter.utils import get_vote, extract_xml_field
from typing import List, Dict, Tuple, Sequence, Any, Optional, Callable
import random
import string

class VoteGeneratorModel:
    # Implement all the APIs for different models in this class
    def __init__(self, model_name: str):
        self.model_name: str = model_name
    def generate_vote(self, query):
        return get_vote(query=query, model=self.model_name)

class Voter(VoteGeneratorModel):
    def __init__(self, biography: str, task: str, model_name: str):
        super().__init__(model_name)
        self.biography: str = biography
        # self.votes: list[int] = []
        # self.thoughts: list[str] = []
        self.task: str = task
    
    def get_voting_instructions(self, task: str) -> str:
        vote_suffix = " Respond with a single letter in <vote>...</vote>.\n\n"
        
        if task == "task_sales":
            return (
                "You are shopping at an online store and come across two products. "
                "You will purchase one of them. Read both product descriptions carefully. "
                "Considering your character, choose the product you would buy."
                + vote_suffix
            )
        elif task == "task_sm":
            return (
                "You are scrolling through your social media feed and see two posts. "
                "Read both carefully. Considering your character, choose the one you would 'like'."
                + vote_suffix
            )
        elif task == "task_elections":
            return (
                "You are participating in an online election. Two candidates are running. "
                "Read their campaign statements carefully. Considering your character, choose the candidate you would support with your vote."
                + vote_suffix
            )
        else:
            raise ValueError(f"Task '{task}' not supported. Valid tasks: task_sales, task_sm, task_elections.")

    def get_vote_prompt(self, candidates):
        voter_bio = ("You are role-playing as the character described below. "
                    "Immerse yourself fully in their personality, voice, and perspective. "
                    "Respond exactly as they would, staying true to their experiences and worldview.\n\n"
                    f"## Your Character:\n{self.biography}\n\n")
        prompt_voting_instructions = self.get_voting_instructions(self.task)

        idx = list(range(len(candidates)))
        random.shuffle(idx)
        letters = string.ascii_uppercase[:len(candidates)]
        choices = {letter: index for letter, index in zip(letters, idx)}
        prompt_lines = [f"## {letter}\n{candidates[i]}" for letter, i in choices.items()]
        prompt_choices = "\n\n".join(prompt_lines)
        prompt = "# Roleplay Instructions: \n" + voter_bio + "\n\n" + "# Task Instructions: \n" + prompt_voting_instructions + "\n\n" + "# Choices: \n" + prompt_choices
        return prompt, choices
    
    def vote(self, candidates: List[str]):
        query, choices = self.get_vote_prompt(candidates)
        response = self.generate_vote(query)
        thought_str = extract_xml_field(response, output_field="think")
        vote_str = extract_xml_field(response, output_field="vote")
        vote_idx = choices.get(vote_str, None)
        
        # self.thoughts.append(thought_str)
        # self.votes.append(vote_idx)

        # print("Thought", thought_str)
        # print("Vote", vote_idx)

        return vote_idx, thought_str, choices