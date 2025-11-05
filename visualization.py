import matplotlib.pyplot as plt
from policy.base import BasePolicy
from agent.core import CoreAgent
from config.train import TrainingConfig
import numpy as np
import os
import datetime
import json

class GraphPloter():
    def __init__(self):
        pass
    
    def plot_eval_rwd_mean(eval_mean_list):
        """
        Plot evaluation reward mean.
        按照时间序列绘制给定的 evaluation reward mean 总体曲线
        """
        # [WriteCode]
        

    def plot_eval_rwd_var(eval_var_list):
        """
        Plot evaluation reward variance.
        按照时间序列绘制给定的 evaluation reward variance 曲线
        """
        # [WriteCode]



    def plot_smoothed_training_rwd(train_rwd_list, window_size=20):
        """
        Plot smoothed training rewards using a moving average.
        按照时间序列绘制给定的 training rewards 曲线
        这个反映训练过程中的 [current_time - window_size, current_time) 内平均值曲线的状态
        随着训练进行，窗口进行移动， 能让 eval_rwd_var 的变化更加平滑
        """
        # [WriteCode]

class Comparator():
    def __init__(self, policy1: BasePolicy, policy2: BasePolicy):
        pass
    
    def run(self):
        """
        Run comparison between two policies.
        比较两个策略的表现
        """
        # [WriteCode]