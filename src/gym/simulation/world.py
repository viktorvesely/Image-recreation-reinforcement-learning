import math
import numpy as np
from enum import Enum
import datetime

from vector import Vector
from particle import Particle
from food import Food
from model.DQN import DQN

settings = {
    "hidden_units": [64, 64], # hidden layers
    "gamma": 0.99, # time reward punishment
    "copy_step": 25, # each n-step transfer the learning from TrainNet to TargetNet,
    "max_experiences": 5000, # max length of the replay vector PER 1 AGENT ONLY!  
    "min_experiences": 100, # min length of the replay vector PER 1 AGENT ONLY!
    "batch_size": 32, # training size per one step
    "num_episodes": 50, # number of episode of the training
    "steps_per_episode": 5000, # number of steps per episode
    "food_distance_limit": 100, # limit how far can each particle see a food
    "initial_epsilon": 0.99, # initial exploration chance
    "epsilon_decay": 0.9999, # After each episode: epsilon *= epsilon_decay 
    "min_epsilon": 0.08  # minimal value for epsilon
}

class World:

    nParticlesPerLine = 30
    lineSize = 60
    frictionCoefficient = 0.1
    updateConsoleTime = 10000

    def __init__(self, num_population, num_food, map_width, map_height):
        self.num_population = num_population
        self.num_food = num_food
        self.map_size = Vector(map_width, map_height)
        self.max_lines = math.floor(self.map_size.y / World.lineSize)
        self.particles_offset = self.map_size.x / World.nParticlesPerLine
        self.population = []
        self.food = []
        self.steps = 0
        self.state_space_n = len(World.state_example())
        self.actions_space_n = len(World.action_example())
        self.total_rewards = np.zeros(settings["num_episodes"])
        self.epsilon = settings["initial_epsilon"]
        self.update_console = 0

        self.TrainNet = DQN(            # This NN collects experiences from multiple agents
            self.state_space_n,   # trains itself and then transfers its learning to target net
            self.actions_space_n,  # Also generates the decisions of the agents
            settings["hidden_units"],
            settings["gamma"],
            settings["max_experiences"],
            settings["min_experiences"],
            settings["batch_size"],
            num_population
            )
        self.TargetNet = DQN(           # This NN recieves weights and biasses from Train net
            self.state_space_n,   # and is used also for the training predictions
            self.actions_space_n,
            settings["hidden_units"],
            settings["gamma"],
            settings["max_experiences"],
            settings["min_experiences"],
            settings["batch_size"],
            num_population
            )

        self.init_food()
        self.init_population()

    class action_space(Enum):
        left = 0
        right = 1
        up = 2
        down = 3
    
    class state_space(Enum):
        foodX = 0
        foodY = 1
        b_food = 2
        enemyX = 3
        enemyY = 4

    @staticmethod
    def random_meaningful_action(): # meaningful means that it it either (left xor right) or/and (top xor down)
        action = World.action_zeros()
        p_left = np.random.random()
        p_up = np.random.random()
        actions = World.action_space()

        if p_left > 0.5:
            action[actions.left] = 1
        else:
            action[actions.right] = 1
        if p_up > 0.5:
            action[actions.up] = 1
        else:
            action[actions.down] = 1
        
        return action
        
         

    @staticmethod
    def action_example_annotated():
        return {
            "left": 0, # go left
            "right": 0, # go right (cancels out with left)
            "up": 0,
            "down": 0
        }

    @staticmethod
    def state_example_annotated():
        return {
            "foodX": 0.0, # 1 / (delta x food pos) 
            "foodY": 0.0, # 1 / (delta y food pos)
            "food_exist": 0, # 0 if there is no food left
            "enemyX": 0.0, # 1 / (delta x enemy pos)
            "enemyY": 0.0 # 1 / (delta y enemy pos)
        }

    @staticmethod
    def action_zeros():
        return np.zeros(4)

    @staticmethod
    def state_zeros():
        return np.zeros(5)
    

    def init_food(self):
        n = self.num_food
        positions = np.random.rand((n, 2))
        positions[:,0] = positions[:,0] * self.map_size.x # scale food x position according to the map width
        positions[:,1] = positions[:,1] * self.map_size.y # scale food y position according to the map height 
        
        self.food = []
        for i in range(n):
            pos = positions[i]
            self.food.append(Food(
                Vector(pos[0], pos[1])
            ))
    
    def get_action(self, state):
        self.TrainNet.get_action(state)

    def init_population(self):
        n = self.num_population
        lines = math.ceil(n / World.nParticlesPerLine)
        if lines > self.max_lines: 
            n = self.max_lines * World.nParticlesPerLine
    
        for i in range(n):
            currentLine = math.floor(i / World.nParticlesPerLine) + 1
            currentIndex = i % World.nParticlesPerLine
            XPos = (currentIndex * self.particles_offset) + (self.particles_offset / 2)
            YPos = currentLine * World.lineSize
            self.population.append(Particle(
                self.TrainNet.get_agent(i),
                self.TrainNet,
                Vector(XPos, YPos)
            ))

    def reset(self):
        self.init_food()
        self.steps = 0
        return #state

    @staticmethod
    def get_now():
        now = datetime.now()
        return datetime.timestamp(now)

    def main_loop(self):
        for n in range(settings["num_episodes"]):
            total_reward = self.episode()
            self.epsilon = max(settings["min_epsilon"], self.epsilon * settings["epsilon_decay"])
            self.total_rewards[n] = total_reward
            # fix variable names and then episode and step function
            avg_rewards = self.total_rewards[max(0, n - 100):(n + 1)].mean()
            now = World.get_now()
            if now - self.update_console > World.updateConsoleTime:
                self.update_console = now
                print("Progress:", int(n/N*100), "%% episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)
    print("avg reward for last 100 episodes:", avg_rewards)

    def episode(self):
        rewards = 0
        done = False
    
        self.reset()
        while self.steps <= settings["steps_per_episode"]:
            action = self.TrainNet.get_action(observations, epsilon, World.random_meaningful_action)
            prev_observations = observations
            self.step(action)
            rewards += reward

            TrainNet.add_experience(exp)
            TrainNet.train(TargetNet)
            iter += 1
            if iter % copy_step == 0:
                TargetNet.copy_weights(TrainNet)
        return rewards  

    def train(self):
        self.TrainNet.train(self.TargetNet)

    def step(self):
        reset = False
        if (self.steps == settings["steps_per_episode"]):
            reset = True

        for particle in self.population:
            state = World.state_zeros()
            action = particle
            for other on self.population:
                
        
        return reset  
    