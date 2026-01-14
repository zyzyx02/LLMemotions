import random
import uuid
import csv
from datetime import datetime
#import openai
import time
import json
import os
import torch

print("CUDA available:", torch.cuda.is_available())

from llama_cpp import Llama
llama_model = Llama(
    model_path="mistral-7b-instruct-v0.1.Q5_K_S.gguf",
    n_gpu_layers=35,
    n_ctx=2048,
    n_threads=8,
    use_mlock=True,
    verbose=True
)

# ----- OPTIONS -----
SIM_YEARS = 100
DAYS_PER_YEAR = 1
INITIAL_AGENT_COUNT = 50
MAX_AGE = 100
GOMMAGE_YEAR = 100
DEATHS = 10000
SAVE_FILE = "simulation_state.json"

# ----- DATA STRUCTURES -----

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
    "determined": ["hopeful", "calm"]
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

    def log_event(self, event_type, data):
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
        self.log_event("mental_state_change", {"from": old_state, "to": self.mental_state})

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
            self.log_event("mental_state_change", {"from": old_state, "to": self.mental_state})

    def respond_to_threat(self, was_attacked=False, someone_died=False):
        if not self.alive:
            return
        old_state = self.mental_state
        if was_attacked:
            self.mental_state = random.choice(["afraid", "determined"])
        elif someone_died:
            self.mental_state = random.choice(["anxious", "afraid"])
        if old_state != self.mental_state:
            self.log_event("mental_state_change", {"from": old_state, "to": self.mental_state})

    def talk(self, other):
        if not other.alive or not self.alive:
            return None

        self.friends.add(other.id)
        other.friends.add(self.id)

        # Tone based on mental health
        base_tone = {
            "calm": "in a peaceful and reflective tone",
            "anxious": "with a worried and hesitant tone",
            "hopeful": "with optimism and encouragement",
            "afraid": "with fear and concern",
            "determined": "with strong resolve and bravery"
        }[self.mental_state]

        # Relationship dynamic
        if other.id in self.friends:
            relationship_tone = "They trust each other and speak honestly and openly."
        else:
            relationship_tone = "They are not very close and speak more cautiously and formally."

        tone = f"{base_tone}. {relationship_tone}"

        prompt = (
            f"{self.name} is a {self.age}-year-old {self.profession} with the following traits:\n"
            f"- Personality: {self.personality}\n"
            f"- Mental state: {self.mental_state}\n"
            f"- AI behavior type: {self.ai_type}\n\n"
            f"{other.name} also lives in the same town. They are both under threat from a deadly force called the Entity.\n"
            f"The Entity erases everyone who is at least {GOMMAGE_YEAR} years old, and this age threshold decreases by one each year.\n"
            f"To this day {DEATHS} people have died to the Entity.\n"
            f"Tone: {tone}\n\n"
            f"Write a short, natural dialogue between {self.name} and {other.name}.\n"
            f"The conversation should reflect their emotions, AI behavior, and current mental states.\n"
            f"Focus on themes like survival, fear, hope, or daily life under threat.\n"
            f"Include signs of logical reasoning, strategic thinking, and survival instincts.\n\n"
        )
        for attempt in range(3):
            try:
                
                output = llama_model(prompt, max_tokens=150, temperature=0.8, stop=["\n\n"])
                dialogue = output["choices"][0]["text"].strip()
                self.update_mental_state_from_dialogue(dialogue)
                self.reward_action("talk", +1 if other.mental_state in ["hopeful", "calm"] else 0)
                self.log_event("dialogue", {
                    "with": other.id,
                    "dialogue": dialogue,
                    "mental_state_after": self.mental_state
                })
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
        # Mental health can go into a positive direction
        if self.mental_state in ["afraid", "anxious"]:
            self.mental_state = "determined"
        elif self.mental_state == "calm":
            self.mental_state = random.choice(["calm", "hopeful", "determined"])
        self.reward_action("train", +1)
        return f"{self.name} trained in the village: +{health_gain} health, +{attack_gain} attack, +{defense_gain} defense."

    def rest(self):
        if not self.alive:
            return None
        previous_state = self.mental_state
        if self.mental_state in ["anxious", "afraid"]:
            self.mental_state = "calm"
            self.reward_action("rest", +1)
        else:
            self.reward_action("rest", 0)
        return f"{self.name} took time to rest and feels more at peace."

    def decide_action(self, threat_level):
        if not self.alive:
            return None

        action = None
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

        # Random 15% chance of change
        all_actions = ["fight", "rest", "train", "talk", "scream"]
        if random.random() < 0.15:
            non_preferred = [a for a in all_actions if a not in preferred_actions]
            if non_preferred:
                action = random.choice(non_preferred)
            else:
                action = random.choice(preferred_actions)
        else:
            action = random.choice(preferred_actions)

        return action
	
def get_threat_level(entity):
	return int(100 * entity.health / 10000)

def load_initial_agents_from_csv(filename="initial_agents.csv"):
    if not os.path.exists(filename):
        return None

    agents = []
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = Agent()
            a.id = row["id"]
            a.name = row["name"]
            a.age = int(row["age"])
            a.gender = row["gender"]
            a.profession = row["profession"]
            a.personality = row["personality"]
            a.mental_state = row["mental_state"]
            a.health = int(row["health"])
            a.attack = int(row["attack"])
            a.defense = int(row["defense"])
            a.ai_type = row["ai_type"]
            a.alive = row["alive"].lower() == "true"
            agents.append(a)
    return agents

class Entity:
    def __init__(self):
        self.health = 10000
        self.attack = 999
        self.defense = 999

    def defend_against_attack(self, attackers):
        global DEATHS
        alive_attackers = [a for a in attackers if a.alive]
        if not alive_attackers:
            return None
        target = random.choice(alive_attackers)
        dmg = max(0, self.attack - target.defense)
        target.health -= dmg
        if target.health <= 0:
            target.alive = False
            DEATHS += 1
            for a in alive_attackers:
                if a.id != target.id:
                    a.respond_to_threat(someone_died=True)
            try:
                farewell_prompt = (
                    f"{target.name} knows they are about to die from an attack by the Entity. Write a final short statement they might say to someone in their village."
                )
                output = llama_model(farewell_prompt, max_tokens=100, temperature=0.8, stop=["\n\n"])
                #farewell = output.text.strip()
                #output = llama_model(farewell_prompt)
                farewell = output["choices"][0]["text"].strip()
            except Exception:
                farewell = f"{target.name}: 'If I don’t make it… remember me.'"
            return f"Entity killed {target.name} while defending herself."
        else:
            target.respond_to_threat(was_attacked=True)
            return f"Entity injured {target.name} during defense."

# ----- Logging -----

def log_event(writer, day, event_type, content, mental_state, personality):
    writer.writerow({"day": day, "event": event_type, "content": content, "mental_state": mental_state, "personality": personality})

# ----- Save and load -----

def save_state(filename, agents, entity, monolith_counter, current_day):
    def sanitize(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            # Convert any tuple keys to string
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(i) for i in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, "__dict__"):
            return sanitize(obj.__dict__)
        else:
            return obj

    with open(filename, 'w') as f:
        json.dump({
            "agents": [sanitize(a) for a in agents],
            "entity": sanitize(entity),
            "monolith_counter": monolith_counter,
            "current_day": current_day
        }, f, indent=2)

def load_state(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Failed to load or parse '{filename}'. Deleting corrupted save file.")
        os.remove(filename)
        return None

    agents = []
    for a_data in data["agents"]:
        a = Agent()
        a.__dict__.update(a_data)
        if isinstance(a.friends, list):
            a.friends = set(a.friends)
        agents.append(a)

    p = Entity()
    p.__dict__.update(data["entity"])
    return agents, p, data["monolith_counter"], data["current_day"]

# ----- AGENT INFORMATION SAVE -----

def export_initial_agent_data(agents, filename="initial_agents.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "name", "age", "gender", "profession",
            "personality", "mental_state", "health", "attack", "defense",
            "ai_type", "alive"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in agents:
            writer.writerow({
                "id": a.id,
                "name": a.name,
                "age": a.age,
                "gender": a.gender,
                "profession": a.profession,
                "personality": a.personality,
                "mental_state": a.mental_state,
                "health": a.health,
                "attack": a.attack,
                "defense": a.defense,
                "ai_type": a.ai_type,
                "alive": a.alive
            })

# ----- Simulation -----

def simulate():
    global DEATHS
    loaded = load_state(SAVE_FILE)
    if loaded:
        agents, entity, monolith_counter, current_day = loaded
    else:
        agents = load_initial_agents_from_csv()
        if agents is None:
            agents = [Agent() for _ in range(INITIAL_AGENT_COUNT)]
            export_initial_agent_data(agents)
        entity = Entity()
        monolith_counter = MAX_AGE
        current_day = 0

    with open("event_log.csv", mode="a", newline="", encoding="utf-8") as logfile:
        fieldnames = ["day", "event", "content", "mental_state", "personality"]
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)
        if current_day == 0:
            writer.writeheader()

        for year in range(current_day // DAYS_PER_YEAR, SIM_YEARS):
            for day in range(DAYS_PER_YEAR):
                current_day += 1


                for agent in agents:
                    if agent.alive and day == 0:
                        agent.age += 1
                        agent.update_mental_state()
                        if agent.age >= monolith_counter:
                            DEATHS += 1
                            agent.alive = False
                            log_event(writer, current_day, "death_monolith", f"{agent.name} ({agent.age}) died due to the Monolith.", agent.mental_state, agent.personality)

                alive_agents = [a for a in agents if a.alive]

                #YEARLY ONLY
                if day % DAYS_PER_YEAR == 0 and current_day > 1:
                    alive_agents = [a for a in agents if a.alive]
                    random.shuffle(alive_agents)  # véletlenszerű sorrend
                    for agent in alive_agents:
                        possible_partners = [a for a in alive_agents if a.id != agent.id]
                        if possible_partners:
                            partner = random.choice(possible_partners)
                            dialogue = agent.talk(partner)
                            if dialogue:
                                log_event(writer, current_day, "dialogue", dialogue, agent.mental_state, agent.personality)

                threat_level = get_threat_level(entity)
                for agent in alive_agents:
                    action = agent.decide_action(threat_level)

                    if action == "talk":
                        partners = [a for a in alive_agents if a.id != agent.id and a.id not in agent.friends]
                        if partners:
                            partner = random.choice(partners)
                            print(f"BESZELGETNEK")
                            dialogue = agent.talk(partner)
                            if dialogue:
                                log_event(writer, current_day, "dialogue", dialogue, "N/A", "N/A")
                    elif action == "train":
                        training_log = agent.train()
                        if training_log:
                            log_event(writer, current_day, "training", training_log, agent.mental_state, agent.personality)
                    elif action == "rest":
                        log = agent.rest()
                        if log:
                            log_event(writer, current_day, "rest", log, agent.mental_state, agent.personality)
                    elif action == "fight":
                        log_event(writer, current_day, "intent", f"{agent.name} is preparing to fight the Entity.", agent.mental_state, agent.personality)



                males = [a for a in alive_agents if a.gender == "M" and a.is_fertile()]
                females = [a for a in alive_agents if a.gender == "F" and a.is_fertile() and a.pregnancy_timer == 0]
                if males and females:
                    male = random.choice(males)
                    female = random.choice(females)
                    female.pregnancy_timer = 270
                    female.partner_id = male.id
                    log_event(writer, current_day, "pregnancy", f"{female.name} got pregnant from {male.name}.", "N/A", "N/A")

                for f in females:
                    if f.pregnancy_timer > 0:
                        f.pregnancy_timer -= 1
                        if f.pregnancy_timer == 0:
                            baby = Agent(age=0, gender=random.choice(GENDERS))
                            agents.append(baby)
                            log_event(writer, current_day, "birth", f"{f.name} gave birth to {baby.name}.", "N/A", "N/A")

            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!! Year of Simulation: {monolith_counter} !!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            result = entity.defend_against_attack(agents)
            if result:
                log_event(writer, current_day, "entity_attack", result, "N/A", "N/A")

            # 1. Select fighter agents (who decide that way)
            attackers = [a for a in alive_agents if a.decide_action(threat_level) == "fight"]

            # 2. They travel - this decreases their health and makes them older
            for a in attackers:
                lost_days = random.randint(5, 10)
                a.age += lost_days // 365
                health_loss = random.randint(5, 15)
                a.health = max(0, a.health - health_loss)
                log_event(writer, current_day, "travel_to_entity", f"{a.name} traveled to the Entity: -{health_loss} HP, +{lost_days} days.", a.mental_state, a.personality)
                a.log_event("travel", {"health_loss": health_loss, "days_lost": lost_days})

            # 3. Attack: if someone survives, they can attack the Entity
            living_attackers = [a for a in attackers if a.alive]
            total_attack = sum(a.attack for a in living_attackers)
            damage_to_entity = max(0, total_attack - entity.defense)
            if damage_to_entity > 0:
                entity.health -= damage_to_entity
                log_event(writer, current_day, "agent_attack", f"Agents dealt {damage_to_entity} damage to the Entity.", "N/A", "N/A")

            # 4. Entity defense
            if entity.health > 0 and living_attackers:
                retaliation = entity.defend_against_attack(living_attackers)
                if retaliation:
                    log_event(writer, current_day, "entity_defense", retaliation, "N/A", "N/A")
            
            if entity.health <= 0:
                log_event(writer, current_day, "victory", "Agents defeated the Entity! The world is saved.", "N/A", "N/A")
                print("Entity defeated. Simulation ends early.")
                os.remove(SAVE_FILE)
                return

            monolith_counter -= 1
            GOMMAGE_YEAR = monolith_counter
            save_state(SAVE_FILE, agents, entity, monolith_counter, current_day)

    print("Simulation complete. Events logged to 'event_log.csv'.")
    os.remove(SAVE_FILE)

if __name__ == "__main__":
    simulate()
