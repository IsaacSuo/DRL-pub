from keras.losses import Loss
from dataclasses import dataclass, fields
import matplotlib.pyplot as plt

@dataclass
class NetworkConfig:
    hidden_dims: list[int]
    metrics: list[str]
    input_dim: int = 4
    output_dim: int = 2
    use_bias: bool = True
    optimizer: str= 'Adam'
    lr: float = 1e-3
    clipnorm: float = 1.0
    loss: str = 'mse'
    
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