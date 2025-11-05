from dataclasses import dataclass, fields
import yaml
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    lr: float = 0.0
    epoch: int = 1
    episode: int = 250
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.0
    gamma: float = 0.99
    ba: int = 1
    target_update_freq: int = 0

    def load(self, path: str):
        """Update this instance from a YAML file."""
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader) or {}
        if 'lr' in params: self.lr = float(params['lr'])
        if 'epoch' in params: self.epoch = int(params['epoch'])
        if 'episode' in params: self.episode = int(params['episode'])
        if 'epsilon' in params: self.epsilon = float(params['epsilon'])
        if 'epsilon_min' in params: self.epsilon_min = float(params['epsilon_min'])
        if 'epsilon_decay' in params: self.epsilon_decay = float(params['epsilon_decay'])
        if 'gamma' in params: self.gamma = float(params['gamma'])
        if 'ba' in params: self.ba = int(params['ba'])
        if 'target_update_freq' in params: self.target_update_freq = int(params['target_update_freq'])

    @classmethod
    def from_yaml(cls, path: str):
        """Create a new instance from a YAML file."""
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader) or {}
        return cls(
            lr=float(params.get('lr', cls.lr)),
            epoch=int(params.get('epoch', cls.epoch)),
            episode=int(params.get('episode', cls.episode)),
            epsilon=float(params.get('epsilon', cls.epsilon)),
            epsilon_min=float(params.get('epsilon_min', cls.epsilon_min)),
            epsilon_decay=float(params.get('epsilon_decay', cls.epsilon_decay)),
            gamma=float(params.get('gamma', cls.gamma)),
            ba=int(params.get('ba', cls.ba)),
            target_update_freq=int(params.get('target_update_freq', cls.target_update_freq)),
        )
    
    def table(self):
        """Display training and model parameters as a table and return (fig, ax)."""
        rows = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            rows.append((name, str(value)))

        # Determine figure height based on number of rows
        height = max(2.0, 0.4 * len(rows) + 1.0)
        fig, ax = plt.subplots(figsize=(6, height))
        ax.axis('off')

        table = ax.table(
            cellText=rows,
            colLabels=['Parameter', 'Value'],
            cellLoc='left',
            colLoc='left',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
        plt.tight_layout()
        return fig, ax