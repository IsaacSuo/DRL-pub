import matplotlib.pyplot as plt
from policy.base import BasePolicy
from config.train_cfg import TrainingConfig
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

class VisualResult():
    def __init__(self, policy: BasePolicy, train_cfg: TrainingConfig, model_cfg):
        self.policy = policy
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
    
    def figure_of_result(self, path=None):
        """Generate figures for training results."""
        # create a single figure with 2x2 subplots
        fig = plt.figure(constrained_layout=True, figsize=(14, 10))
        axs = [fig.add_subplot(2, 2, i + 1) for i in range(4)]

        # create the original individual figures/axes
        fig0, ax0 = self.table_hyperparameters()
        fig1, ax1 = self.table_model_architecture()
        fig2, ax2 = self.plot_smoothed_training_rwd()
        fig3, ax3 = self.plot_eval_rwd_var()

        def reparent_artists(old_ax, new_ax):
            # move all artists from old_ax to new_ax
            for artist in list(old_ax.get_children()):
            # skip axis background artists that new_ax already has
                if artist in (old_ax.xaxis, old_ax.yaxis):
                    continue
                try:
                    # detach from old axes (if possible)
                    old_ax._children.remove(artist)
                except Exception:
                    pass
                try:
                    # try to bind artist to new axes
                    artist.set_axes(new_ax)
                except Exception:
                    try:
                        artist.set_parent(new_ax)
                    except Exception:
                        pass
                try:
                    new_ax.add_artist(artist)
                except Exception:
                    # last resort: ignore artists that cannot be moved
                    pass

                # copy basic axis labels/titles/limits if present
                try:
                    new_ax.set_title(old_ax.get_title())
                    new_ax.set_xlabel(old_ax.get_xlabel())
                    new_ax.set_ylabel(old_ax.get_ylabel())
                    new_ax.set_xlim(old_ax.get_xlim())
                    new_ax.set_ylim(old_ax.get_ylim())
                except Exception:
                    pass

        reparent_artists(ax0, axs[0])
        reparent_artists(ax1, axs[1])
        reparent_artists(ax2, axs[2])
        reparent_artists(ax3, axs[3])

        # close the original standalone figures to avoid duplicates
        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
        if path is not None:
            fig.savefig(path)
        return fig, axs
                 
    def plot_smoothed_training_rwd(self, window_size=20):
        """Plot smoothed training rewards using a moving average."""
        arr = np.asarray(self.policy.train_reward_lst, dtype=float)
        window_size = max(1, int(window_size))
        if window_size > arr.size:
            window_size = arr.size
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(arr, window, mode='valid')

        x_raw = np.arange(arr.size)
        x_smooth = np.arange(window_size - 1, arr.size)
        
        fig, ax = plt.subplots()
        ax.plot(x_raw, arr, alpha=0.3, label='raw reward')
        ax.plot(x_smooth, smoothed, color='C1', label=f'smoothed (w={window_size})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward (raw and smoothed)')
        ax.legend()
        ax.grid(True)
        return fig, ax
    
    def plot_eval_rwd_var(self):
        """Plot evaluation reward variance."""
        arr = np.asarray(self.policy.eval_reward_var_lst, dtype=float)
        x_raw = np.arange(arr.size)
        fig, ax = plt.subplots()
        ax.plot(x_raw, arr, alpha=0.3, label='reward variance')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Variance')
        ax.set_title('Training Reward Variance')
        ax.legend()
        ax.grid(True)
        return fig, ax
        
    def table_hyperparameters(self):
        """Display training and model parameters."""
        try:
            params = vars(self.train_cfg)
        except TypeError:
            # fallback if vars() fails
            try:
                params = dict(self.train_cfg)
            except Exception:
                params = getattr(self.train_cfg, "__dict__", {}) or {}

        # prepare rows and stringify values (truncate long values)
        max_val_len = 100
        rows = []
        for k, v in params.items():
            if isinstance(v, dict):
                val = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
            else:
                val = str(v)
            if len(val) > max_val_len:
                val = val[:max_val_len - 3] + "..."
            rows.append((str(k), val))

        # build table
        n_rows = max(1, len(rows))
        fig, ax = plt.subplots(figsize=(8, 0.4 * n_rows + 1.2))
        ax.axis("off")
        col_labels = ["Parameter", "Value"]
        cell_text = [[k, v] for k, v in rows]

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        plt.title("Training Configuration Parameters")
        plt.tight_layout()
        return fig, ax
    
    def table_model_architecture(self):
        """Display model architecture summary."""
        try:
            params = vars(self.model_cfg)
        except TypeError:
            try:
                params = dict(self.model_cfg)
            except Exception:
                params = getattr(self.model_cfg, "__dict__", {}) or {}

        # prepare rows and stringify values (handle nested structures and truncate)
        max_val_len = 200
        rows = []
        for k, v in params.items():
            if isinstance(v, dict):
                try:
                    val = json.dumps(v, ensure_ascii=False)
                except Exception:
                    val = str(v)
            elif isinstance(v, (list, tuple, set)):
                try:
                    val = ", ".join(map(str, v))
                except Exception:
                    val = str(v)
            else:
                val = str(v)

            if len(val) > max_val_len:
                val = val[:max_val_len - 3] + "..."
            rows.append((str(k), val))

        # build table
        n_rows = max(1, len(rows))
        fig, ax = plt.subplots(figsize=(10, 0.4 * n_rows + 1.2))
        ax.axis("off")
        col_labels = ["Field", "Value"]
        cell_text = [[k, v] for k, v in rows]

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        plt.title("Model Configuration")
        plt.tight_layout()
        return fig, ax
    