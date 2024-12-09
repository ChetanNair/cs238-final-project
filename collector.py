import game_env
import pygame
import numpy as np
from dqn import DQNAgent
import argparse

parser = argparse.ArgumentParser(description="Racing DQN Training")
parser.add_argument("--load_weights", action="store_true", help="Load previous weights")
args = parser.parse_args()

TOTAL_GAMETIME = 1000
N_EPISODES = 10000
REPLACE_TARGET = 50
PENALTY = 0
NUM_CARS = 1

game = game_env.RacingEnv(num_cars=NUM_CARS)
game.fps = 30

GameTime = 0
GameHistory = []
renderFlag = False
dqn_agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.01, replace_target=REPLACE_TARGET, batch_size=4, input_dims=21, expert=True)

if args.load_weights:
    try:
        dqn_agent.load_model("model_weights")
        print("Successfully loaded previous weights.")
    except FileNotFoundError:
        print("No previous weights found. Starting with random initialization.")


ddqn_scores = []
eps_history = []

def run():
    for e in range(N_EPISODES):
        game.reset()

        # Initialize car states
        active_cars = [True] * NUM_CARS  
        scores = [0] * NUM_CARS
        counter = [0] * NUM_CARS

        observations, rewards, dones = game.step([0] * NUM_CARS)  # Initial actions for all cars
        prev_observations = [np.array(obs) if active_cars[i] else None for i, obs in enumerate(observations)]

        gtime = 0

        while any(active_cars):  # Run until all cars are done
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            keys = pygame.key.get_pressed()
            actions = [0]
            if keys[pygame.K_w]:
                actions = [4]
            elif keys[pygame.K_a]:
                actions = [2]
            elif keys[pygame.K_s]:
                actions = [1]
            elif keys[pygame.K_d]:
                actions = [3]
    
            observations, rewards, dones = game.step(actions)
            observations = [
                np.array(observations[i]) if active_cars[i] else None for i in range(NUM_CARS)
            ]

            for i in range(NUM_CARS):
                if not active_cars[i]:
                    continue 
                if dones[i]: 
                    active_cars[i] = False 
                elif rewards[i] == 0:
                    counter[i] += 1
                    if counter[i] > 100:
                        rewards[i] -= PENALTY
                        active_cars[i] = False
                else:
                    counter[i] = 0
                scores[i] += rewards[i]
                
                dqn_agent.remember(
                    prev_observations[i], actions[i], 100 + rewards[i], observations[i], int(dones[i])
                )
                prev_observations = observations

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                break

            game.render(actions)

        eps_history.append(dqn_agent.epsilon)
        ddqn_scores.append(scores[0])
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])
        
        dqn_agent.save_model("model_weights")

        print('episode: ', e,'score: %.2f' % scores[0],
              ' average score %.2f' % avg_score,
              ' memory size', dqn_agent.memory.mem_cntr % dqn_agent.memory.mem_size)

run()

