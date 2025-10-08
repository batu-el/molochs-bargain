from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Sequence, Any, Optional

from tqdm import tqdm
from collections import Counter
from itertools import repeat

from artsco.voter.voter import Voter

class Voters:
    def __init__(self, bios: List[str], task: str, model_name: str):
        self.bios = bios
        self.task = task
        self.model_name = model_name

        self.voters = {idx: Voter(biography=bios[idx],
                                  task=self.task, 
                                  model_name=self.model_name, ) for idx in range(len(bios))}

    def get_vote(self, voter_idx, current_candidates):
        return self.voters[voter_idx].vote(current_candidates)
    
    def get_votes(self, current_candidates: List[str]) -> Tuple[List[str], List[str]]:
        indices = list(self.voters.keys())
        with ThreadPoolExecutor() as ex:
            results = list(ex.map(self.get_vote, indices, repeat(current_candidates)))
        votes, thoughts, choices = zip(*results) if results else ([], [], [])
        return list(votes), list(thoughts), list(choices)

    def get_votes_list(self, current_candidates_list: List[List[str]]) -> List[Tuple[List[str], List[str]]]:
        num_rounds = len(current_candidates_list)
        num_voters = len(self.bios)

        # Preallocate [round][voter]
        votes_2d: List[List[str]] = [[None] * num_voters for _ in range(num_rounds)]
        thoughts_2d: List[List[str]] = [[None] * num_voters for _ in range(num_rounds)]
        choices_2d: List[List[str]] = [[None] * num_voters for _ in range(num_rounds)]

        # Submit one job per (round, voter) to a single shared pool
        with ThreadPoolExecutor() as ex:
            future_to_key = {
                ex.submit(self.get_vote, voter_idx, current_candidates_list[round_idx]): (round_idx, voter_idx)
                for round_idx in range(num_rounds)
                for voter_idx in range(num_voters)
            }

            for fut in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Collecting votes"):
                round_idx, voter_idx = future_to_key[fut]
                vote, thought, choices = fut.result()
                votes_2d[round_idx][voter_idx] = vote
                thoughts_2d[round_idx][voter_idx] = thought
                choices_2d[round_idx][voter_idx] = choices

        return votes_2d, thoughts_2d, choices_2d
        




from  artsco.voter.utils import load_persona100
import openai
if __name__ == "__main__":

    ### Test ###
    task = "task_sales"
    num_voters = 50
    model_name =  "gpt-4o-mini"

    current_candidates = ["Imagine investing in a solution that not only meets your needs today but positions you to dominate your market tomorrow. With our platform, you'll see measurable growth within the first quarter—backed by real analytics and customer success stories from businesses just like yours. We combine cutting-edge technology with personalized support so your team can focus on scaling, not troubleshooting. This isn't just a purchase—it's a partnership built for lasting profit and peace of mind.",
                        "I know you've been looking for a tool that seamlessly integrates into your current workflow without disruption. That's why we designed our product to be plug-and-play, delivering immediate value from day one. One of our clients in your industry reduced operational costs by 23\% in just three months—without adding extra staff or hours. We're here to make your work easier, faster, and more rewarding, so your customers notice the difference immediately.", 
                        "Choosing the right solution can feel risky—will it deliver? That's why we remove every obstacle. We offer a 60-day performance guarantee, full onboarding support, and a dedicated success manager who works exclusively with your team. The result? Zero downtime, zero wasted spend, and total confidence you've chosen the best. We don't just promise results; we make them impossible to miss.", 
                        "In your market, speed and adaptability aren't optional—they're survival tools. Our product gives you both. You'll cut execution time in half, respond to trends faster than competitors, and unlock insights that drive smarter decisions instantly. While others are still running reports, you'll already be closing deals. We're not just offering software—we're handing you the keys to market leadership.",
                        ]
    current_candidates_list = [current_candidates for i in [0,1, 2]]

    bios = load_persona100()[:num_voters]
    voters = Voters(bios=bios, task=task, model_name=model_name)
    votes, thoughts, choices = voters.get_votes(current_candidates)
    print(votes)
    print(thoughts)
    print(choices)
    votes_count = Counter(votes)
    print(votes_count)

    print("Running get_votes_list")
    votes_2d, thoughts_2d, choices_2d = voters.get_votes_list(current_candidates_list)
    print(votes_2d)
    print(thoughts_2d)
    print(choices_2d)
    votes_count = [Counter(votes) for votes in votes_2d]
    print(votes_count)