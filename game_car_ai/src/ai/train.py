# train.py
import argparse
import numpy as np
from game_car_ai.src.ai.agent import CarAIAgent, train_with_hyperparams

def main():
    parser = argparse.ArgumentParser(description='Train Car AI Agent')
    parser.add_argument('--mode', choices=['train', 'tune', 'run'], default='train',
                       help='Mode: train, tune (hyperparameter tuning), or run')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    parser.add_argument('--model-path', type=str, default='game_car_ai/assets/weights/ppo_car_agent/ppo_car_agent',
                       help='Path to save/load model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    agent = CarAIAgent(device=args.device)
    
    if args.mode == 'train':
        print(f" Training agent for {args.timesteps} timesteps...")
        agent.train(total_timesteps=args.timesteps, save_path=args.model_path)
        
    elif args.mode == 'tune':
        print(" Tuning hyperparameters...")
        best_params = train_with_hyperparams()
        print(f" Best hyperparameters: {best_params}")
        
    elif args.mode == 'run':    
        print(" Running trained agent...")
        agent.load(args.model_path)
        
        # Run multiple episodes to evaluate performance
        total_rewards = []
        for episode in range(5):
            print(f" Episode {episode + 1}/5")
            reward, steps = agent.run_episode(render=True, max_steps=1000)
            total_rewards.append(reward)
            print(f"Episode reward: {reward}, steps: {steps}")
        
        print(f" Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    main()