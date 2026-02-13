import torch.nn as nn
from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/custom")
class CustomBackbone(AbstractBackbone):
    """Пример backbone для новой модальности.
    
    Замени на свой Transformer, GNN, 1D-UNet и т.д.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.get("in_dim", 512), 1024),
            nn.GELU(),
            nn.Linear(1024, config.get("out_dim", 512))
        )
    
    def _forward_impl(self, x, timestep, condition=None, **kwargs):
        # timestep можно использовать как conditioning
        return self.net(x)