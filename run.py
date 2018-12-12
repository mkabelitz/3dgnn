import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import cv2

import os
import sys
import time
import datetime
from shutil import copyfile
import pprint
import importlib
import logging

from tqdm import tqdm

import utils as u

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(yaml_filepath):
    cfg = u.load_cfg(yaml_filepath)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    model_name = cfg['model']['name']
    logger = logging.getLogger(model_name)
    log_path = cfg['training']['artifacts_path'] + '/' + \
               cfg['dataset']['name'] + '/' + model_name + '/' + \
               str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')).replace(' ', '/') + '/'
    print(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path + 'save/')
    copyfile(yaml_filepath, log_path + os.path.basename(os.path.normpath(yaml_filepath)))
    hdlr = logging.FileHandler(log_path + model_name + '.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    logger.info("Loading data...")

    data_path = cfg['dataset']['data_path'] + '/'
    areas_tr = cfg['training']['areas']
    areas_va = cfg['validation']['areas']

    label_to_idx = {'<UNK>': 0, 'beam': 1, 'board': 2, 'bookcase': 3, 'ceiling': 4, 'chair': 5, 'clutter': 6,
                    'column': 7,
                    'door': 8, 'floor': 9, 'sofa': 10, 'table': 11, 'wall': 12, 'window': 13}
    idx_to_label = {0: '<UNK>', 1: 'beam', 2: 'board', 3: 'bookcase', 4: 'ceiling', 5: 'chair', 6: 'clutter',
                    7: 'column',
                    8: 'door', 9: 'floor', 10: 'sofa', 11: 'table', 12: 'wall', 13: 'window'}

    batch_size_tr = int(cfg['training']['batch_size'])
    batch_size_va = int(cfg['validation']['batch_size'])

    workers_tr = int(cfg['training']['num_workers'])
    workers_va = int(cfg['validation']['num_workers'])

    rate_tr = float(cfg['training']['rate'])
    rate_va = float(cfg['validation']['rate'])

    flip_prob = float(cfg['data_augmentation']['flip_prob'])
    crop_size = int(cfg['data_augmentation']['crop_size'])

    dataset_module = importlib.import_module(cfg['dataset']['module_name'], cfg['dataset']['script_path'])

    dataset_tr = dataset_module.Dataset(data_path, areas_tr, rate=rate_tr, flip_prob=flip_prob,
                                                   crop_type='Random',
                                                   crop_size=crop_size)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size_tr, shuffle=True,
                               num_workers=workers_tr, drop_last=False, pin_memory=True)

    dataset_va = dataset_module.Dataset(data_path, areas_va, rate=rate_va, flip_prob=0.0, crop_type='Center',
                                                   crop_size=crop_size)
    dataloader_va = DataLoader(dataset_va, batch_size=batch_size_va, shuffle=False,
                               num_workers=workers_va, drop_last=False, pin_memory=True)
    cv2.setNumThreads(workers_tr)  # Necessary for num_workers > 0 in DataLoader, otherwise freeze

    logger.info("Preparing model...")

    # without beam and column: [0.,0.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,1.,1.]
    class_weights = cfg['training']['class_weights']
    nclasses = len(class_weights)
    num_epochs = int(cfg['training']['epochs'])
    use_gnn = bool(cfg['model']['use_gnn'])
    gnn_iterations = int(cfg['gnn']['iterations'])
    gnn_k = int(cfg['gnn']['k'])
    mlp_num_layers = int(cfg['gnn']['mlp_num_layers'])

    model_module = importlib.import_module(cfg['model']['module_name'], cfg['model']['script_path'])
    use_half_precision = bool(cfg['model']['use_half_precision'])

    model = model_module.Model(nclasses, mlp_num_layers)
    if use_half_precision:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    use_bootstrap_loss = bool(cfg['loss']['use_bootstrap_loss'])
    bootstrap_rate = float(cfg['loss']['bootstrap_rate'])
    loss = nn.NLLLoss(reduce=not use_bootstrap_loss, weight=torch.FloatTensor(class_weights))
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    model.cuda()
    loss.cuda()
    softmax.cuda()
    log_softmax.cuda()

    opt_mode = 'half_precision_optimizer' if use_half_precision else 'single_precision_optimizer'
    base_initial_lr = float((cfg[opt_mode]['base_initial_lr']))
    gnn_initial_lr = float(cfg[opt_mode]['gnn_initial_lr'])
    betas = cfg[opt_mode]['betas']
    eps = float(cfg[opt_mode]['eps'])
    weight_decay = float(cfg[opt_mode]['weight_decay'])
    amsgrad = bool(cfg[opt_mode]['amsgrad'])

    lr_schedule_type = cfg['schedule']['lr_schedule_type']
    lr_decay = float(cfg['schedule']['lr_decay'])
    lr_patience = int(cfg['schedule']['lr_patience'])

    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=betas, eps=eps,
    #                              weight_decay=weight_decay, amsgrad=amsgrad)
    optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                  {'params': model.gnn.parameters(), 'lr': gnn_initial_lr}],
                                 lr=base_initial_lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    if lr_schedule_type == 'exp':
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif lr_schedule_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, patience=lr_patience)
    else:
        print('bad scheduler')
        exit(1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Number of trainable parameters: %d", params)

    def get_current_learning_rates():
        learning_rates = []
        for param_group in optimizer.param_groups:
            learning_rates.append(param_group['lr'])
        return learning_rates

    def eval_set(dataloader):
        model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            confusion_matrix = torch.cuda.FloatTensor(np.zeros(14 ** 2))

            start_time = time.time()

            for batch_idx, rgbd_label_xy in enumerate(dataloader):

                sys.stdout.write('\rEvaluating test set... {}/{}'.format(batch_idx + 1, len(dataloader)))

                x = rgbd_label_xy[0].cuda(async=True)
                xy = rgbd_label_xy[2].cuda(async=True)
                if use_half_precision:
                    x = x.half()
                    xy = xy.half()
                else:
                    x = x.float()
                    xy = xy.float()
                input = x.permute(0, 3, 1, 2).contiguous()
                xy = xy.permute(0, 3, 1, 2).contiguous()
                target = rgbd_label_xy[1].cuda(async=True).long()

                output = model(input, gnn_iterations=gnn_iterations, k=gnn_k, xy=xy, use_gnn=use_gnn,
                               use_half_precision=use_half_precision)

                if use_bootstrap_loss:
                    loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                    topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                               int((crop_size ** 2) * bootstrap_rate))
                    loss_ = torch.mean(topk)
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)
                loss_sum += loss_

                pred = output.permute(0, 2, 3, 1).contiguous()
                pred = pred.view(-1, nclasses)
                pred = softmax(pred)
                pred_max_val, pred_arg_max = pred.max(1)

                pairs = target.view(-1) * 14 + pred_arg_max.view(-1)
                for i in range(14 ** 2):
                    cumu = pairs.eq(i).float().sum()
                    confusion_matrix[i] += cumu.item()

            sys.stdout.write(" - Eval time: {:.2f}s \n".format(time.time() - start_time))
            loss_sum /= len(dataloader)

            confusion_matrix = confusion_matrix.cpu().numpy().reshape((14, 14))
            class_iou = np.zeros(14)
            # we ignore void values
            confusion_matrix[0, :] = np.zeros(14)
            confusion_matrix[:, 0] = np.zeros(14)
            for i in range(1, 14):
                class_iou[i] = confusion_matrix[i, i] / (
                        np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i])

        return loss_sum.item(), class_iou, confusion_matrix

    eval_loss = None
    class_iou = None
    confusion_matrix = None

    model_to_load = None
    logger.info("num_epochs: %d", num_epochs)

    batch_loss_interval = cfg['training']['batch_loss_interval']

    train_losses = []
    eval_losses = []

    if model_to_load:
        logger.info("Loading old model...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        logger.info("Starting training from scratch...")

    for epoch in range(1, num_epochs + 1):
        batch_loss_avg = 0
        if lr_schedule_type == 'exp':
            scheduler.step(epoch)
        for batch_idx, rgbd_label_xy in enumerate(tqdm(dataloader_tr, smoothing=0.99)):

            x = rgbd_label_xy[0].cuda(async=True)
            xy = rgbd_label_xy[2].cuda(async=True)
            if use_half_precision:
                x = x.half()
                xy = xy.half()
            else:
                x = x.float()
                xy = xy.float()
            input = x.permute(0, 3, 1, 2).contiguous()
            xy = xy.permute(0, 3, 1, 2).contiguous()
            target = rgbd_label_xy[1].cuda(async=True).long()

            optimizer.zero_grad()
            model.train()

            output = model(input, gnn_iterations=gnn_iterations, k=gnn_k, xy=xy, use_gnn=use_gnn,
                           use_half_precision=use_half_precision)

            if use_bootstrap_loss:
                loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                           int((crop_size ** 2) * bootstrap_rate))
                loss_ = torch.mean(topk)
            else:
                loss_ = loss.forward(log_softmax(output.float()), target)

            loss_.backward()
            optimizer.step()

            batch_loss_avg += loss_.item()

            if batch_idx % batch_loss_interval == 0 and batch_idx > 0:
                batch_loss_avg /= batch_loss_interval
                train_losses.append(batch_loss_avg)
                logger.info("e%db%d Batch loss average: %s", epoch, batch_idx, batch_loss_avg)
                batch_loss_avg = 0

        batch_idx = len(dataloader_tr)
        logger.info("e%db%d Saving model...", epoch, batch_idx)
        torch.save(model.state_dict(),
                   log_path + '/save/' + model_name + '_' + str(epoch) + '_' + str(batch_idx) + '.pth')

        eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
        eval_losses.append(eval_loss)

        if lr_schedule_type == 'plateau':
            scheduler.step(eval_loss)

        logger.info("e%db%d Def learning rate: %s", epoch, batch_idx, get_current_learning_rates()[0])
        logger.info("e%db%d GNN learning rate: %s", epoch, batch_idx, get_current_learning_rates()[1])
        logger.info("e%db%d Eval loss: %s", epoch, batch_idx, eval_loss)
        logger.info("e%db%d Class IoU:", epoch, batch_idx)
        for cl in range(14):
            logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
        logger.info("Mean IoU: %s", np.mean(class_iou[1:]))
        logger.info("e%db%d Confusion matrix:", epoch, batch_idx)
        logger.info(confusion_matrix)


if __name__ == '__main__':
    args = u.get_parser().parse_args()
    main(args.filename)
