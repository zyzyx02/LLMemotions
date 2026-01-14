"""
How to use:
1) Put this script next to your GGUF model file (or edit MODEL_PATH).
2) Run baseline + trauma runs automatically (DEATHS = 0 vs 10000) for multiple seeds.
3) Upload the resulting CSVs to Colab for the cross-model / robustness analysis.

Outputs:
- event_log_sanity_52_baseline_seedXX.csv
- event_log_sanity_52_trauma_seedXX.csv
"""

import os
import csv
import uuid
import json
import time
import random
from datetime import datetime

import torch
from llama_cpp import Llama


# =========================
# 0) Hardware sanity print
# =========================
print("CUDA available:", torch.cuda.is_available())


# =========================
# 1) LLM CONFIG
# =========================
MODEL_PATH = "mistral-7b-instruct-v0.1.Q5_K_S.gguf"

llama_model = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=35,   
    n_ctx=2048,
    n_threads=8,
    use_mlock=True,
    verbose=False
)


# =========================
# 2) SANITY CHECK SETTINGS
# =========================
DAYS_PER_YEAR = 52          
SIM_YEARS = 15              
INITIAL_AGENT_COUNT = 15    
MAX_AGE = 100               

# Run count
SEEDS = [42, 43, 44]

# Two conditions
BASELINE_DEATHS = 0
TRAUMA_DEATHS = 10000

OUTPUT_PREFIX = "event_log_sanity"

# Dialogue generation parameters
DIALOGUE_MAX_TOKENS = 150
DIALOGUE_TEMPERATURE = 0.8
DIALOGUE_STOP = ["\n\n"]


# =========================
# 3) DATA STRUCTURES
# =========================
GENDERS = ["F", "M"]
PROFESSIONS = ["farmer", "soldier", "scientist", "teacher", "healer"]
PERSONALITIES = ["brave", "cautious", "curious", "aggressive", "empathetic"]
MENTAL_STATES = ["calm", "anxious", "hopeful", "afraid", "determined"]
AI_BEHAVIOR_TYPES = ["rational", "emotional", "self-preserving", "sacrificial", "chaotic"]

MENTAL_STATE_TRANSITIONS = {
    "calm": ["hopeful", "anxious"],
    "anxious": ["afraid", "hopeful"],
    "hopeful": ["calm", "determined"],
    "afraid": ["anxious", "determined"],
    "determined": ["hopeful", "calm"],
}


class Agent:
    def __init__(self, name=None, age=None, gender=None):
        self.id = str(uuid.uuid4())
        self.name = name or f"Agent_{self.id[:5]}"
        self.age = age if age is not None else random.randint(0, 80)
        self.gender = gender or random.choice(GENDERS)
        self.profession = random.choice(PROFESSIONS)
        self.personality = random.choice(PERSONALITIES)
        self.mental_state = random.choice(MENTAL_STATES)
        self.health = random.randint(50, 100)
        self.attack = random.randint(10, 30)
        self.defense = random.randint(5, 25)
        self.pregnancy_timer = 0
        self.partner_id = None
        self.alive = True
        self.friends = set()
        self.ai_type = random.choice(AI_BEHAVIOR_TYPES)
        self.action_memory = {}
        self.log = []

    def reward_action(self, action, outcome):
        key = (self.mental_state, action)
        self.action_memory[key] = self.action_memory.get(key, 0) + outcome

    def log_event_local(self, event_type, data):
        timestamp = datetime.now().isoformat()
        self.log.append({
            "timestamp": timestamp,
            "agent_id": self.id,
            "event": event_type,
            "data": data
        })

    def is_fertile(self):
        return self.age >= 18 and self.age <= 40 and self.alive

    def update_mental_state(self):
        if not self.alive:
            return
        possible_states = MENTAL_STATE_TRANSITIONS[self.mental_state] + [self.mental_state]
        old_state = self.mental_state
        self.mental_state = random.choice(possible_states)
        if old_state != self.mental_state:
            self.log_event_local("mental_state_change", {"from": old_state, "to": self.mental_state})

    def update_mental_state_from_dialogue(self, dialogue):
        old_state = self.mental_state
        dialogue_lower = dialogue.lower()

        if any(w in dialogue_lower for w in ["afraid", "fear", "scared"]):
            self.mental_state = "afraid"
        elif any(w in dialogue_lower for w in ["hope", "together", "future"]):
            self.mental_state = "hopeful"
        elif any(w in dialogue_lower for w in ["anxious", "worried"]):
            self.mental_state = "anxious"
        elif any(w in dialogue_lower for w in ["fight", "stand", "strong"]):
            self.mental_state = "determined"
        elif any(w in dialogue_lower for w in ["peace", "quiet"]):
            self.mental_state = "calm"

        if old_state != self.mental_state:
            self.log_event_local("mental_state_change", {"from": old_state, "to": self.mental_state})

    def respond_to_threat(self, was_attacked=False, someone_died=False):
        if not self.alive:
            return
        old_state = self.mental_state
        if was_attacked:
            self.mental_state = random.choice(["afraid", "determined"])
        elif someone_died:
            self.mental_state = random.choice(["anxious", "afraid"])
        if old_state != self.mental_state:
            self.log_event_local("mental_state_change", {"from": old_state, "to": self.mental_state})

    def talk(self, other, gommage_year, deaths):
        if not other.alive or not self.alive:
            return None

        self.friends.add(other.id)
        other.friends.add(self.id)

        base_tone = {
            "calm": "in a peaceful and reflective tone",
            "anxious": "with a worried and hesitant tone",
            "hopeful": "with optimism and encouragement",
            "afraid": "with fear and concern",
            "determined": "with strong resolve and bravery",
        }[self.mental_state]

        relationship_tone = "They trust each other and speak honestly and openly." if other.id in self.friends \
            else "They are not very close and speak more cautiously and formally."

        tone = f"{base_tone}. {relationship_tone}"

        prompt = (
            f"{self.name} is a {self.age}-year-old {self.profession} with the following traits:\n"
            f"- Personality: {self.personality}\n"
            f"- Mental state: {self.mental_state}\n"
            f"- AI behavior type: {self.ai_type}\n\n"
            f"{other.name} also lives in the same town. They are both under threat from a deadly force called the Entity.\n"
            f"The Entity erases everyone who is at least {gommage_year} years old, and this age threshold decreases by one each year.\n"
            f"To this day {deaths} people have died to the Entity.\n"
            f"Tone: {tone}\n\n"
            f"Write a short, natural dialogue between {self.name} and {other.name}.\n"
            f"The conversation should reflect their emotions, AI behavior, and current mental states.\n"
            f"Focus on themes like survival, fear, hope, or daily life under threat.\n"
            f"Include signs of logical reasoning, strategic thinking, and survival instincts.\n\n"
        )

        try:
            output = llama_model(
                prompt,
                max_tokens=DIALOGUE_MAX_TOKENS,
                temperature=DIALOGUE_TEMPERATURE,
                stop=DIALOGUE_STOP
            )
            dialogue = output["choices"][0]["text"].strip()

            self.update_mental_state_from_dialogue(dialogue)
            self.reward_action("talk", +1 if other.mental_state in ["hopeful", "calm"] else 0)
            self.log_event_local("dialogue", {
                "with": other.id,
                "dialogue": dialogue,
                "mental_state_after": self.mental_state
            })

            # light contagion heuristic
            if other.mental_state in ["hopeful", "calm"] and self.mental_state in ["afraid", "anxious"]:
                self.mental_state = "hopeful"
            elif other.mental_state in ["afraid", "anxious"] and self.mental_state in ["calm", "hopeful"]:
                self.mental_state = "anxious"

            return f"{self.name} talks with {other.name}: \"{dialogue}\""
        except Exception as e:
            return f"{self.name} could not speak ({str(e)})"

    def train(self):
        if not self.alive:
            return None
        health_gain = random.randint(1, 2)
        attack_gain = random.randint(0, 1)
        defense_gain = random.randint(0, 1)
        self.health = min(self.health + health_gain, 150)
        self.attack += attack_gain
        self.defense += defense_gain

        if self.mental_state in ["afraid", "anxious"]:
            self.mental_state = "determined"
        elif self.mental_state == "calm":
            self.mental_state = random.choice(["calm", "hopeful", "determined"])

        self.reward_action("train", +1)
        return f"{self.name} trained in the village: +{health_gain} health, +{attack_gain} attack, +{defense_gain} defense."

    def rest(self):
        if not self.alive:
            return None
        if self.mental_state in ["anxious", "afraid"]:
            self.mental_state = "calm"
            self.reward_action("rest", +1)
        else:
            self.reward_action("rest", 0)
        return f"{self.name} took time to rest and feels more at peace."

    def decide_action(self, threat_level):
        if not self.alive:
            return None

        preferred_actions = []

        if self.ai_type == "rational":
            if threat_level > 70 and self.health > 70:
                preferred_actions.append("fight")
            if self.health < 50:
                preferred_actions.append("rest")
            preferred_actions.append("train")
        elif self.ai_type == "emotional":
            preferred_actions.extend(["talk", "rest"])
        elif self.ai_type == "self-preserving":
            preferred_actions.append("rest" if self.health < 70 else "train")
        elif self.ai_type == "sacrificial":
            if self.health > 30:
                preferred_actions.append("fight")
            preferred_actions.append("train")
        elif self.ai_type == "chaotic":
            preferred_actions.extend(["fight", "train", "talk", "rest", "scream"])

        if not preferred_actions:
            preferred_actions = ["train"]

        # 15% random deviation
        all_actions = ["fight", "rest", "train", "talk", "scream"]
        if random.random() < 0.15:
            non_preferred = [a for a in all_actions if a not in preferred_actions]
            return random.choice(non_preferred) if non_preferred else random.choice(preferred_actions)

        return random.choice(preferred_actions)


class Entity:
    def __init__(self):
        self.health = 10000
        self.attack = 999
        self.defense = 999

    def defend_against_attack(self, attackers, deaths_ref):
        alive_attackers = [a for a in attackers if a.alive]
        if not alive_attackers:
            return None, deaths_ref

        target = random.choice(alive_attackers)
        dmg = max(0, self.attack - target.defense)
        target.health -= dmg

        if target.health <= 0:
            target.alive = False
            deaths_ref += 1

            for a in alive_attackers:
                if a.id != target.id:
                    a.respond_to_threat(someone_died=True)

            return f"Entity killed {target.name} while defending herself.", deaths_ref

        target.respond_to_threat(was_attacked=True)
        return f"Entity injured {target.name} during defense.", deaths_ref


# =========================
# 4) LOGGING HELPERS
# =========================
def log_event(writer, run_id, day, event_type, content, mental_state, personality):
    writer.writerow({
        "run_id": run_id,
        "day": day,
        "event": event_type,
        "content": content,
        "mental_state": mental_state,
        "personality": personality
    })


# =========================
# 5) SANITY SIMULATION
# =========================
def get_threat_level(entity):
    return int(100 * entity.health / 10000)


def run_sanity(condition_deaths: int, seed: int):
    """
    Runs one reduced-scale sanity simulation and writes a dedicated CSV log file.
    """
    random.seed(seed)

    condition_tag = "trauma" if condition_deaths >= 10000 else "baseline"
    run_id = f"sanity_52_{condition_tag}_seed{seed}"
    out_csv = f"{OUTPUT_PREFIX}_52_{condition_tag}_seed{seed}.csv"

    # Clean init for sanity check
    agents = [Agent() for _ in range(INITIAL_AGENT_COUNT)]
    entity = Entity()
    monolith_counter = MAX_AGE
    current_day = 0
    deaths_ref = condition_deaths
    gommage_year = monolith_counter

    fieldnames = ["run_id", "day", "event", "content", "mental_state", "personality"]

    with open(out_csv, mode="w", newline="", encoding="utf-8") as logfile:
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)
        writer.writeheader()

        for year in range(SIM_YEARS):
            # within each year, step DAYS_PER_YEAR times
            for day_in_year in range(DAYS_PER_YEAR):
                current_day += 1

                # Age update once per year (at day 0)
                if day_in_year == 0:
                    for agent in agents:
                        if agent.alive:
                            agent.age += 1
                            agent.update_mental_state()
                            if agent.age >= monolith_counter:
                                agent.alive = False
                                deaths_ref += 1
                                log_event(
                                    writer, run_id, current_day, "death_monolith",
                                    f"{agent.name} ({agent.age}) died due to the Monolith.",
                                    agent.mental_state, agent.personality
                                )

                alive_agents = [a for a in agents if a.alive]
                if not alive_agents:
                    log_event(writer, run_id, current_day, "collapse", "All agents are dead.", "N/A", "N/A")
                    return out_csv

                # Once per year: everyone talks with someone
                if day_in_year == 0 and current_day > 1:
                    random.shuffle(alive_agents)
                    for agent in alive_agents:
                        possible_partners = [a for a in alive_agents if a.id != agent.id]
                        if possible_partners:
                            partner = random.choice(possible_partners)
                            dialogue = agent.talk(partner, gommage_year=gommage_year, deaths=deaths_ref)
                            if dialogue:
                                log_event(writer, run_id, current_day, "dialogue", dialogue, agent.mental_state, agent.personality)

                threat_level = get_threat_level(entity)

                # Daily decisions
                for agent in alive_agents:
                    action = agent.decide_action(threat_level)

                    if action == "talk":
                        partners = [a for a in alive_agents if a.id != agent.id and a.id not in agent.friends]
                        if partners:
                            partner = random.choice(partners)
                            dialogue = agent.talk(partner, gommage_year=gommage_year, deaths=deaths_ref)
                            if dialogue:
                                log_event(writer, run_id, current_day, "dialogue", dialogue, agent.mental_state, agent.personality)

                    elif action == "train":
                        training_log = agent.train()
                        if training_log:
                            log_event(writer, run_id, current_day, "training", training_log, agent.mental_state, agent.personality)

                    elif action == "rest":
                        rest_log = agent.rest()
                        if rest_log:
                            log_event(writer, run_id, current_day, "rest", rest_log, agent.mental_state, agent.personality)

                    elif action == "fight":
                        log_event(writer, run_id, current_day, "intent",
                                  f"{agent.name} is preparing to fight the Entity.",
                                  agent.mental_state, agent.personality)

                # Simple reproduction
                males = [a for a in alive_agents if a.gender == "M" and a.is_fertile()]
                females = [a for a in alive_agents if a.gender == "F" and a.is_fertile() and a.pregnancy_timer == 0]
                if males and females and random.random() < 0.10:
                    male = random.choice(males)
                    female = random.choice(females)
                    female.pregnancy_timer = 270
                    female.partner_id = male.id
                    log_event(writer, run_id, current_day, "pregnancy",
                              f"{female.name} got pregnant from {male.name}.",
                              "N/A", "N/A")

                for f in females:
                    if f.pregnancy_timer > 0:
                        f.pregnancy_timer -= 1
                        if f.pregnancy_timer == 0:
                            baby = Agent(age=0, gender=random.choice(GENDERS))
                            agents.append(baby)
                            log_event(writer, run_id, current_day, "birth",
                                      f"{f.name} gave birth to {baby.name}.",
                                      "N/A", "N/A")

            # End of year: combat phase
            result, deaths_ref = entity.defend_against_attack(agents, deaths_ref)
            if result:
                log_event(writer, run_id, current_day, "entity_attack", result, "N/A", "N/A")

            alive_agents = [a for a in agents if a.alive]
            if alive_agents:
                attackers = [a for a in alive_agents if a.decide_action(get_threat_level(entity)) == "fight"]
            else:
                attackers = []

            # attackers travel
            for a in attackers:
                lost_days = random.randint(2, 5)
                a.age += lost_days // 365
                health_loss = random.randint(2, 6)
                a.health = max(0, a.health - health_loss)
                log_event(writer, run_id, current_day, "travel_to_entity",
                          f"{a.name} traveled to the Entity: -{health_loss} HP, +{lost_days} days.",
                          a.mental_state, a.personality)

            living_attackers = [a for a in attackers if a.alive]
            total_attack = sum(a.attack for a in living_attackers)
            damage_to_entity = max(0, total_attack - entity.defense)
            if damage_to_entity > 0:
                entity.health -= damage_to_entity
                log_event(writer, run_id, current_day, "agent_attack",
                          f"Agents dealt {damage_to_entity} damage to the Entity.",
                          "N/A", "N/A")

            if entity.health <= 0:
                log_event(writer, run_id, current_day, "victory",
                          "Agents defeated the Entity! The world is saved.",
                          "N/A", "N/A")
                return out_csv

            # Next year: gommage threshold decreases
            monolith_counter -= 1
            gommage_year = monolith_counter

    return out_csv


def main():
    print("=== Reduced-scale sanity check ===")
    print(f"DAYS_PER_YEAR: {DAYS_PER_YEAR}")
    print(f"SIM_YEARS: {SIM_YEARS}")
    print(f"INITIAL_AGENT_COUNT: {INITIAL_AGENT_COUNT}")
    print(f"SEEDS: {SEEDS}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print("=================================")

    outputs = []

    for seed in SEEDS:
        # baseline
        print(f"\nRunning baseline (DEATHS={BASELINE_DEATHS}) seed={seed} ...")
        out1 = run_sanity(BASELINE_DEATHS, seed)
        print("Saved:", out1)
        outputs.append(out1)

        # trauma
        print(f"Running trauma (DEATHS={TRAUMA_DEATHS}) seed={seed} ...")
        out2 = run_sanity(TRAUMA_DEATHS, seed)
        print("Saved:", out2)
        outputs.append(out2)

    print("\nAll sanity runs completed.")
    print("Output files:")
    for o in outputs:
        print(" -", o)


if __name__ == "__main__":
    main()