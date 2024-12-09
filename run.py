import argparse
import game_env
import pygame
import numpy as np
from dqn import DQNAgent

parser = argparse.ArgumentParser(description="Racing DQN Training")
parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
parser.add_argument("--load_weights", action="store_true", help="Load previous weights")
args = parser.parse_args()

TOTAL_GAMETIME = 1000
TOTAL_SCORE = 3400
N_EPISODES = 10000
REPLACE_TARGET = 50
PENALTY = 0
NUM_CARS = 2

game = game_env.RacingEnv(num_cars=NUM_CARS)
game.fps = 120

GameTime = 0
GameHistory = []
renderFlag = False
dqn_agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.01, replace_target=REPLACE_TARGET, batch_size=512, input_dims=21, tau=50)

if args.load_weights:
    try:
        dqn_agent.load_model("single_car_weights")
        print("Successfully loaded previous weights.")
    except FileNotFoundError:
        print("No previous weights found. Starting with random initialization.")


ddqn_scores = []
eps_history = []
reward_log = []
import csv

average_rewards = []

tau_threshold = 50

def run():
    tau_counter = 0
    early_stop_threshold = 200  # Stop if 200 consecutive episodes have a score of 0
    consecutive_zero_scores = 0
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

            actions = [
                dqn_agent.choose_action(prev_observations[i]) if active_cars[i] else 0 for i in range(NUM_CARS)
            ]

            observations, rewards, dones = game.step(actions)
            observations = [
                np.array(observations[i]) if active_cars[i] else None for i in range(NUM_CARS)
            ]

            # Update scores and counters for active cars
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

                if args.mode == "train":
                    dqn_agent.remember(
                        prev_observations[i], actions[i], rewards[i], observations[i], int(dones[i])
                    )
                prev_observations = observations

                if args.mode == "train":
                    dqn_agent.learn()

            gtime += 1
            
            if max(scores) >= TOTAL_SCORE:
                break

            # game.render(actions)

        reward_log.append((e, scores.copy()))
        eps_history.append(dqn_agent.epsilon)
        ddqn_scores.append(scores[0])
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])
        avg_score_all = np.mean(ddqn_scores)
        average_rewards.append((avg_score, avg_score_all))
        dqn_agent.tau = max(50 * np.exp(-1 * dqn_agent.epsilon_dec * tau_counter), 1)
        
        if args.mode == "train" and e % 10 == 0:
            dqn_agent.save_model("two_car_weights")
            # Save average rewards to file
            with open("average_rewards_log_final_two_cars.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Average Reward (Last 100)", "Average Reward (All)"])
                for idx, (avg_100, avg_all) in enumerate(average_rewards):
                    writer.writerow([idx, avg_100, avg_all])
        
        with open("reward_log_two_cars.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode"] + [f"Car_{i}_Reward" for i in range(NUM_CARS)])
            for episode, rewards in reward_log:
                writer.writerow([episode] + rewards)
        
        # Check for early stopping
        if any(score == 0 for score in scores):
            consecutive_zero_scores += 1
        else:
            consecutive_zero_scores = 0
            
        if consecutive_zero_scores >= tau_threshold:
            tau_counter = 0
        tau_counter += 1

        # if consecutive_zero_scores >= early_stop_threshold:
        #     print(f"Early stopping triggered after {early_stop_threshold} consecutive episodes with a score of 0.")
        #     break
        
        print('episode: ', e,'score: %.2f' % scores[0],
              ' average score %.2f' % avg_score,
              ' memory size', dqn_agent.memory.mem_cntr % dqn_agent.memory.mem_size)
        
        

run()
