import random

import numpy as np
import torch
import warnings

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,build_runner,get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger

from os.path import exists

from tools.losslandscape import net_plotter
import h5py
import time
import torch.nn as nn
from tools.losslandscape import scheduler
import tools.losslandscape.mpi4pytorch as mpi
import tools.losslandscape.projection as proj
import copy
import numpy as np
from mmcv.runner import load_checkpoint

import os
from tools.losslandscape.plot_hessian_eigen import crunch_hessian_eigs
from tools.losslandscape import plot_2D , plot_1D


def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):

    # Setup env for preventing lock on h5py file for newer h5py versions
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r') #发生异常: OSError Unable to open file (bad object header version number)
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    # xcoordinates = np.linspace(int(args.xmin), int(args.xmax), int(num=args.xnum))
    xcoordinates = np.linspace(int(args.xmin), int(args.xmax), num=int(args.xnum))

    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(int(args.xmin), int(args.xmax), num=int(args.xnum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(runner, surf_file, net, w, s, d, data_loaders, loss_key, loss_key_cls, loss_key_loc, comm, rank, args,cfg,  logger,      
        distributed,
        validate,
        timestamp,
        meta):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    # losses, accuracies = [], []
    losses = []
    losses_cls = []
    losses_loc = []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        losses_cls = -np.ones(shape=shape)
        losses_loc = -np.ones(shape=shape)

        # accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[loss_key_cls] = losses_cls
            f[loss_key_loc] = losses_loc
            # f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        losses_cls = f[loss_key_cls][:]
        losses_loc = f[loss_key_loc][:]
        # accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    logger.info('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    # criterion = nn.CrossEntropyLoss()
    # if args.loss_name == 'mse':
    #     criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    # for count, ind in enumerate(inds):
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            # print("set_weights前:")
            # print(net.module.parameters())
            # before_set_weights = net.module.parameters()
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
            # print("set_weights后:")
            # print(net.module.parameters())
            # after_set_weights = net.module.parameters()
            # diff = after_set_weights - before_set_weights
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        # need to rewrite the follow line for object detection, to need loss, I need to use the train.py 
        # loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        
        runner.model = net

        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
        
        loss = runner.total_loss
        loss_loc = runner.total_loss_loc
        loss_cls = runner.total_loss_cls

        runner._epoch -= 1 

        # loss = train_detector(
        #     net,
        #     dataloader,
        #     cfg,
        #     distributed=distributed,
        #     validate=(not args.no_validate),
        #     timestamp=timestamp,
        #     meta=meta)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        losses_cls.ravel()[ind] = loss_cls
        losses_loc.ravel()[ind] = loss_loc
        # accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses    = mpi.reduce_max(comm, losses)
        losses_cls    = mpi.reduce_max(comm, losses_cls)
        losses_loc    = mpi.reduce_max(comm, losses_loc)
        # accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[loss_key_cls][:] = losses_cls
            f[loss_key_loc][:] = losses_loc
            # f[acc_key][:] = accuracies
            f.flush()

        # print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
        #         rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
        #         acc_key, acc, loss_compute_time, syc_time))
        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.3f \t%s=%.3f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss, loss_key_cls, loss_cls, loss_key_loc, loss_loc,
                loss_compute_time, syc_time))
        logger.info(f'Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.3f \t%s=%.3f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss, loss_key_cls, loss_cls, loss_key_loc, loss_loc,
                loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        # accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()



def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector_loss_landscape(model,
                   dataset,
                   cfg,
                   args,
                   comm,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)#False
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    

    # build runner
    # auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)
  

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))



    # init runner
    # runner = EpochBasedRunner(
    #     model,
    #     optimizer=optimizer,
    #     work_dir=cfg.work_dir,
    #     logger=logger,
    #     meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    torch.autograd.set_detect_anomaly(True)

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    w = net_plotter.get_weights(model) # initial parameters
    s = copy.deepcopy(model.state_dict()) # deepcopy since state_dict are references    

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    rank, _ = get_dist_info()
    dir_file = net_plotter.name_direction_file(args) # name the direction file    
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, model) #set
    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file) # OSError: Unable to open file (bad object header version number)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

  # load directions
    d = net_plotter.load_directions(dir_file)
    # d = d.cuda(cfg.gpu_ids[0]) 
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)


    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------

    if args.hessian == 'False':
        crunch(runner,surf_file, model, w, s, d, data_loaders, 'train_loss', 'train_loss_cls', 'train_loss_loc', comm, rank, args, cfg, logger,  
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
    else:
        print ("Hessian calculating!")
        crunch_hessian_eigs(runner, surf_file, model, w, s, d, data_loaders, comm,  rank, args,cfg)
        print ("Rank " + str(rank) + ' is done!')

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    # if args.plot and rank == 0:
    if args.plot:
        if args.y:
            plot_2D.plot_2d_eig_ratio(surf_file, 'min_eig', 'max_eig', args.show)
        else:
            plot_1D.plot_1d_eig_ratio(surf_file, args.xmin, args.xmax, 'min_eig', 'max_eig')

    # crunch(dir_file, model, w, s, d, datasets, 'train_loss', 'train_acc', rank, args, cfg,   
    #     distributed=distributed,
    #     validate=(not args.no_validate),
    #     timestamp=timestamp,
    #     meta=meta)




