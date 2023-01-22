import mmcv.runner.epoch_based_runner as runner
from mmcv.runner.builder import RUNNERS
import mmcv
from mmcv.runner.hooks.hook import HOOKS 

import time

@RUNNERS.register_module()
class LossLandscapeRunner(runner.EpochBasedRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self, data_loader, **kwargs):
        # self.optimizer.epoch=self._epoch
        # self.optimizer.max_epochs = self._max_epochs
        # self.optimizer.max_iters = self._max_iters
        # self.optimizer.iter = self._iter
        # runner.EpochBasedRunner.train(self, data_loader, **kwargs)
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # new add
        self.total_loss = 0
        self.total_loss_cls = 0
        self.total_loss_loc = 0
        self.total = 0 # number of samples



        for i, data_batch in enumerate(self.data_loader):
            batch_size=self.data_loader.batch_size
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            # self.call_hook('after_train_iter') # we don't need loss backpropagation
            self._iter += 1
            # new add

            # we don't need to *batch_size
            # self.total_loss += (self.outputs['loss']).item() * batch_size #module 'mmcv.runner.epoch_based_runner' has no attribute 'outputs'

            # self.total_loss_cls += (self.outputs['loss_cls']).item() * batch_size #module 'mmcv.runner.epoch_based_runner' has no attribute 'outputs'
            # self.total_loss_loc += (self.outputs['loss_loc']).item() * batch_size #module 'mmcv.runner.epoch_based_runner' has no attribute 'outputs'

            self.total_loss += (self.outputs['loss']).item() 

            self.total_loss_cls += (self.outputs['loss_cls']).item() 
            self.total_loss_loc += (self.outputs['loss_loc']).item() 
            
            self.total += batch_size
            
        self.total_loss = self.total_loss/self.total

        self.total_loss_cls = self.total_loss_cls/self.total
        self.total_loss_loc = self.total_loss_loc/self.total
        

        # self.call_hook('after_train_epoch')
        self._epoch += 1

        # return self.total_loss

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