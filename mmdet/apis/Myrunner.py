import mmcv.runner.epoch_based_runner as runner
from mmcv.runner.builder import RUNNERS
import mmcv
from mmcv.runner.hooks.hook import HOOKS 



@RUNNERS.register_module()
class BopRunner(runner.EpochBasedRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self, data_loader, **kwargs):
        self.optimizer.epoch=self._epoch
        self.optimizer.max_epochs = self._max_epochs
        self.optimizer.max_iters = self._max_iters
        self.optimizer.iter = self._iter
        runner.EpochBasedRunner.train(self, data_loader, **kwargs)
    "set a new optimizerHook"
    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            if 'type' not in optimizer_config:
                optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)