# src/ai/agent.py
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import gymnasium as gym

class SaveBestModelCallback(BaseCallback):
    """
    Callback choosing the best model base on 'episode reward'.
    """
    def __init__(self, save_path, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.best_mean_reward = -float('inf')
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Kiểm tra ep_info_buffer có dữ liệu
        if len(self.model.ep_info_buffer) > 0:
            # Lấy reward của episode gần nhất
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                mean_reward = ep_info['r']
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_file = os.path.join('game_car_ai/assets/weights/ppo_car_agent_1', 'best_model.zip')
                    self.model.save(save_file)
                    if self.verbose > 0:
                        print(f"\n New best model saved with reward {self.best_mean_reward:.2f}")
        return True  
class CarAIAgent:
    def __init__(self, policy='MlpPolicy', device='auto'):
        self.model = None
        self.env = None
        self.policy = policy
        self.device = device
        
    def create_env(self, n_stack=4):
        """Create and wrap environment"""
        from game_car_ai.src.env.game_env import CarGameEnv
        # Create base environment
        self.env = CarGameEnv()
        
        # Add monitoring
        self.env = Monitor(self.env)
        
        # Vectorize environment (required for SB3)
        self.env = DummyVecEnv([lambda: self.env])

        return self.env
    
    def create_model(self, env=None):
        """Create PPO model with MLP policy"""
        if env is None:
            env = self.create_env()
            
        self.model = PPO(
            self.policy,  # MlpPolicy cho vector inputs
            env,
            verbose=1,
            device=self.device,
            learning_rate=3e-4,
            n_steps=512,  
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64])  # Smaller network
            )
        )
        return self.model
  

    def train(self, total_timesteps=100000, save_path="game_car_ai/assets/weights/ppo_car_agent"):
        """Train the AI agent"""
        print("Starting AI agent training...")
        
        # Create environment
        self.env = self.create_env()
        # Create model
        if self.model is None:
            self.create_model(self.env)
        # Create callback

        callback = SaveBestModelCallback(save_path=save_path)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="ppo_car_driver",
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save the trained model
        self.save(save_path)
        
        
        print(f"Training completed! Model saved to {save_path}")
        return self.model
    
    def save(self, path):
        """Save trained agent"""
        self.model.save(path)
        print(f"Saved agent to {path}")
    
    def load(self, path, env=None):
        """Load trained agent"""
        if self.env is None:
            self.env = self.create_env()
        print(2)
        self.model = PPO.load(path, env=env)
        print(f"Loaded agent from {path}")
        return self.model
    
    def predict(self, observation, deterministic=True):
        """Predict action for given observation"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def run_episode(self, render=True, max_steps=1000):
        """Run a single episode with the trained agent"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        
        if self.env is None:
            self.env = self.create_env()
        
        obs = self.env.reset()
        total_reward = 0
        step_count = 0
        
        print("Running AI agent...")
        
        try:
            for step in range(max_steps):
                action = self.predict(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                
                if render:
                    self.env.render()
                
                if done:
                    print(f" Episode finished after {step_count} steps, total reward: {total_reward:.2f}")
                    break
                    
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            return total_reward, step_count

# Training function with hyperparameter tuning
def train_with_hyperparams():
    """Train with different hyperparameters"""
    agent = CarAIAgent()
    
    # Hyperparameter grid
    learning_rates = [3e-4, 1e-4, 3e-5]
    batch_sizes = [32, 64, 128]
    
    best_reward = -float('inf')
    best_params = {}
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with lr={lr}, batch_size={batch_size}")
            
            # Create new model with current hyperparams
            agent.create_model()
            agent.model.learning_rate = lr
            agent.model.batch_size = batch_size
            
            # Train briefly to evaluate
            agent.train(total_timesteps=10000, 
                       save_path=f"game_car_ai/assets/weights/ppo_car_agent/ppo_lr{lr}_bs{batch_size}")
            
            # Evaluate
            reward, steps = agent.run_episode(render=False, max_steps=500)
            print(f"Evaluation reward: {reward:.2f}, steps: {steps}")
            
            if reward > best_reward:
                best_reward = reward
                best_params = {'lr': lr, 'batch_size': batch_size}
    
    print(f"Best params: {best_params}, reward: {best_reward:.2f}")
    return best_params

if __name__ == "__main__":
    # Quick test
    agent = CarAIAgent()
    env = agent.create_env()
    
    # Test with random actions
    print("Testing environment with random actions...")
    obs = env.reset()
    
    for i in range(50):
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()
            
    env.close()