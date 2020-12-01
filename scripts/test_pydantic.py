import torch
import hydra
from omegaconf import OmegaConf, DictConfig

from seg_lapa.config_parse.train_conf import TrainConf, get_dataconf, get_optimconf


def parse_config(cfg: DictConfig) -> TrainConf:
    config = TrainConf(
        dataset=get_dataconf(cfg.dataset),
        optimizer=get_optimconf(cfg.optimizer)
    )

    return config


class FakeModel(torch.nn.Module):
    def __init__(self):
        super(FakeModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(48, 48, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


@hydra.main(config_path='../seg_lapa/config', config_name='train')
def main(cfg):
    print('\nHydra\'s Config:')
    print(OmegaConf.to_yaml(cfg))

    print('\nParsed Datamodule:')
    config = parse_config(cfg)
    print(config)

    print('\nInitialized Datamodule:')
    print(config.dataset.get_datamodule())

    print('\nInitialized Optimizer:')
    model = FakeModel()
    print(config.optimizer.get_optimizer(model.parameters()))


if __name__ == '__main__':
    main()
