import gymnasium as gym
import keras
import numpy as np
import time 
from typing import Any, Callable

from config.train_cfg import TrainingConfig
from policy.base import BasePolicy, DoneException

class Trainer():
    def __init__(self, 
                 model_dir: str,
                 get_run_logdir: Callable[[], Any],
                 ):
        # Set up environment
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0] # Number of observations (CartPole)
        self.action_size = self.env.action_space.n            # Number of possible actions
        self.cb = keras.callbacks.TensorBoard(log_dir = get_run_logdir(model_dir), histogram_freq=1)
        
        
    def train(self, policy: BasePolicy, cfg: TrainingConfig):
        '''
        组合不同的策略和训练配置进行训练
        默认中每个 baseline approach 都应该使用相同的 TrainingConfig
        '''
        # prepare all materials before training
        policy.prepare(self.env)
        epsilon = cfg.epsilon
        train_counter = 0 # Train Counter for weight syncing
        total_training_time = 0 # For timing training
        
        for ep in range(cfg.episode):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0

            # record start time
            start = time.time()

            for _ in range(500):
                action = policy.act(state, self.action_size, epsilon)                
                state, total_reward, epsilon, done = policy.step(
                    env=self.env,
                    state_size=self.state_size,
                    state=state,
                    action=action,
                    total_reward=total_reward,
                    train_counter=train_counter,
                    cb=self.cb,
                    epsilon=epsilon,
                    ba=cfg.ba,
                    gamma=cfg.gamma,
                    epsilon_min=cfg.epsilon_min,
                    epsilon_decay=cfg.epsilon_decay,
                    target_update_freq=cfg.target_update_freq,
                    epoch=cfg.epoch,
                )
                if done:
                    break
                
            # record end time and log training time
            end = time.time()
            total_training_time += end - start

            # Evaluation
            # [WriteCode]   
            eval_reward_mean, eval_reward_var = policy.evaluate(max_timesteps=500)

            # Log
            policy.log(total_reward, eval_reward_mean, eval_reward_var, ep, cfg.episode, epsilon)

            # Early Stopping Condition to avoid overfitting
            # If the evaluation reward reaches the specified threshold, stop training early.
            # The default threshold is set to 500, but you should adjust this based on observed training performance.
            if policy.early_stopping(ep, eval_reward_mean, threshold=500):
                break

        # record end time and calculate average training time per episode
        # evaluate average training time per episode
        print(f"Training time: {total_training_time/cfg.episode:.4f} seconds per episode")

        self.env.close()

# if __name__ == 'main':

#     from config.train_cfg import TrainingConfig

#     train_cfg = TrainingConfig(
#         lr=lr,
#         epoch=epoch,
#         episode=episode,
#         epsilon=epsilon,
#         epsilon_min=epsilon_min,
#         epsilon_decay=epsilon_decay,
#         gamma=gamma,
#         ba=ba,
#         target_update_freq=target_update_freq
#     )
#     train_cfg.load('config/hyperparams.yml')

#     # DDQN Baseline Model
#     from keras.models import Sequential
#     from keras.layers import Dense
#     from keras.losses import SparseCategoricalCrossentropy, MeanSquaredError

#     # [WriteCode] from ... import ...
#     from policy.ddqn import DoubleDeepQNetworkPolicy
#     from model.ddqn import DoubleDeepQNetworkTagetModel, DoubleDeepQNetworkEvalModel
#     from config.ddqn_cfg import DDQNConfig

#     # Define the eval (online) network
#     ddqn_cfg_eval = DDQNConfig(
#         input_dim=state_size,
#         output_dim=action_size,
#         use_bias=True,
#         is_compiled=True,
#         hidden_dims=[64, 64],
#         optimizer='adam',
#         loss=MeanSquaredError(),
#         metrics=['accuracy'])
#     eval_model = DoubleDeepQNetworkEvalModel(ddqn_cfg_eval)
#     print('eval_model:')
#     eval_model.summary()

#     # Create target_model with the same architecture
#     ddqn_cfg_target = DDQNConfig(
#         input_dim=state_size,
#         output_dim=action_size,
#         use_bias=True,
#         is_compiled=False,
#         hidden_dims=[10, 10],
#         optimizer='adam',
#         loss=MeanSquaredError(),
#         metrics=['accuracy'])
#     target_model = DoubleDeepQNetworkTagetModel(ddqn_cfg_target)
#     print('target_model:')
#     target_model.summary()

#     # Skip compiling as target_model will not be trained with .fit()
#     # Instead, weights will be copied from the online model
#     target_model.set_weights(eval_model.get_weights())
#     xqy_policy = DoubleDeepQNetworkPolicy(
#         target_model=target_model,
#         eval_model=eval_model
#     )

#     # Set up environment
#     env = gym.make("CartPole-v1")
#     state_size = env.observation_space.shape[0] # Number of observations (CartPole)
#     action_size = env.action_space.n            # Number of possible actions
    
#     def get_run_logdir(k):
#         root_logdir = os.path.join(os.curdir, "eec4400_logs", k)
#         run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#         return os.path.join(root_logdir, run_id)
    
#     from trainner import Trainer
#     ddqn_trainner = Trainer(
#         model_dir="ddqn_baseline",
#         get_run_logdir=get_run_logdir,
#     )
#     ddqn_trainner.train(xqy_policy_planb, train_cfg)

#     from visualization import VisualResult
#     res_xqy_policy = VisualResult(xqy_policy, train_cfg, ddqn_cfg_target)
#     _, ax1 = res_xqy_policy_planb.plot_smoothed_training_rwd()
#     _, ax2 = res_xqy_policy_planb.plot_eval_rwd_var()
#     _, ax3 = res_xqy_policy_planb.table_hyperparameters()
#     _, ax4 = res_xqy_policy_planb.table_model_architecture()
#     ax1.save('1.png')
#     ax2.save('2.png')
#     ax3.save('3.png')
#     ax4.save('4.png')