from dataclasses import dataclass, fields
from typing import List
import matplotlib.pyplot as plt


@dataclass
class QNetworkConfig:
    hidden_dims: List[int]
    metrics: List[str]
    input_dim: int = 4
    output_dim: int = 2
    use_bias: bool = True
    optimizer: str = 'Adam'
    lr: float = 1e-4
    clipnorm: float = 1.0
    loss: str = 'mse'

    def table(self):
        rows = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            rows.append((name, str(value)))

        height = max(2.0, 0.45 * len(rows) + 1.0)
        fig, ax = plt.subplots(figsize=(6.5, height))
        ax.axis('off')

        table = ax.table(
            cellText=rows,
            colLabels=['Parameter', 'Value'],
            cellLoc='left',
            colLoc='left',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.3)
        plt.tight_layout()
        return fig, ax