'''Nexus Learning — reinforcement learning, experience replay, reward shaping.'''
import math, random, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

@dataclass
class Experience:
    state: Tuple; action: int; reward: float; next_state: Tuple; done: bool

class QTable:
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 0.1):
        self.q: Dict = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate; self.gamma = discount; self.epsilon = epsilon
    def get(self, state: tuple, action: int) -> float:
        return self.q[state][action]
    def best_action(self, state: tuple) -> int:
        actions = self.q[state]
        if not actions: return 0
        return max(actions, key=actions.get)
    def choose(self, state: tuple, n_actions: int = 4) -> int:
        if random.random() < self.epsilon: return random.randint(0, n_actions-1)
        return self.best_action(state)
    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool) -> float:
        current = self.q[state][action]
        if done: target = reward
        else: target = reward + self.gamma * max(self.q[next_state].values(), default=0)
        td = target - current
        self.q[state][action] += self.lr * td
        return td

class ExperienceReplay:
    def __init__(self, capacity: int = 10000):
        self.buffer: List[Experience] = []; self.capacity = capacity
    def add(self, exp: Experience) -> None:
        self.buffer.append(exp)
        if len(self.buffer) > self.capacity: self.buffer.pop(0)
    def sample(self, batch_size: int = 32) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class RewardShaper:
    def __init__(self):
        self.weights: Dict[str, float] = {}
    def set_weight(self, factor: str, weight: float) -> None:
        self.weights[factor] = weight
    def compute(self, base_reward: float, factors: Dict[str, float]) -> float:
        shaped = base_reward
        for factor, value in factors.items():
            weight = self.weights.get(factor, 0)
            shaped += weight * value
        return shaped

class SkillAcquisition:
    def __init__(self):
        self.skills: Dict[str, Dict] = {}; self.mastery: Dict[str, float] = {}
    def learn(self, skill_name: str, context: Dict, outcome: float) -> float:
        if skill_name not in self.skills:
            self.skills[skill_name] = {"attempts": 0, "successes": 0, "total_reward": 0}
        s = self.skills[skill_name]; s["attempts"] += 1; s["total_reward"] += outcome
        if outcome > 0: s["successes"] += 1
        mastery = s["successes"] / s["attempts"]
        self.mastery[skill_name] = mastery
        return mastery

def demo():
    print("=== Learning ===")
    qt = QTable(epsilon=0.3)
    rewards = 0
    for ep in range(100):
        state = (0, 0)
        for step in range(10):
            action = qt.choose(state, 4)
            reward = 1.0 if action == 2 else -0.1
            next_state = (min(9, state[0]+1), action)
            qt.update(state, action, reward, next_state, step==9)
            state = next_state; rewards += reward
    print(f"Q-learning: 100 eps, {rewards:.0f} total reward")
    print(f"  Best action at (0,0): {qt.best_action((0,0))} (Q={qt.get((0,0), qt.best_action((0,0))):.2f})")
    replay = ExperienceReplay()
    for _ in range(50): replay.add(Experience((0,0), 1, 0.5, (1,0), False))
    batch = replay.sample(5)
    print(f"  Replay buffer: {len(replay.buffer)} exps, sampled {len(batch)}")
    shaper = RewardShaper(); shaper.set_weight("efficiency", 0.5); shaper.set_weight("safety", -2.0)
    r = shaper.compute(1.0, {"efficiency": 0.8, "safety": 0.1})
    print(f"  Shaped reward: {r:.2f} (base=1.0 + efficiency*0.5 + safety*-2.0)")
    skills = SkillAcquisition()
    for _ in range(20): m = skills.learn("survey", {}, 1.0 if random.random()>0.3 else -1.0)
    print(f"  Skill 'survey' mastery: {m:.0%} ({skills.skills['survey']['successes']}/{skills.skills['survey']['attempts']})")

if __name__ == "__main__": demo()
