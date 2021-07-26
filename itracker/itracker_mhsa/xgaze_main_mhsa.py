import sys
sys.path.append('../utils/')
import shutil, time, argparse, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter

from visdom import Visdom 

from pathlib import Path
from torch.nn import init
from loguru import logger
from eva_error import angular_error
from torch.nn.parallel import DistributedDataParallel as DDP


from ITrackerModel_xgaze_mhsa import ITrackerModel
from ITrackData_Xgaze_mhsa import ITrackDataXgaze

parse = argparse.ArgumentParser(description='iTracker-xgaze-trainer')
parse.add_argument('--data_path', default='/home/data/wjc_data/xgaze_224_prepare',
                   help='the train and test dataset path')
parse.add_argument('--load', default=True, help='load the model')
parse.add_argument('--train', default=True, help='train and validate')
args = parse.parse_args()

data_path = args.data_path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

workers = 10
epochs = 25
batch_size = 128

base_lr = 0.0001
momentum = 0.9
weight_decay = 0.0001
print_freq = 10
best_predict = 1e20
current_predict = 0
lr = base_lr

count_train = 0
count_test = 0


def main():
    global args, weight_decay, momentum, current_predict, best_predict

    do_load = args.load
    do_train = args.train
    do_test = not args.train

    ngpus = torch.cuda.device_count()
    world_size = ngpus
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, do_train,do_load, do_test))


    # model.apply(weight_init)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # model = torch.nn.DataParallel(model)


def main_worker(pid, ngpus, do_train,do_load, do_test):
    args.gpu = pid
    args.rank = pid
    print(pid)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=ngpus, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    model = ITrackerModel()
    model.cuda(args.gpu)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    imageSize = (224, 224)
    cudnn.benchmark = True

    load_epoch = 0

    # criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': base_lr}], lr,
                                 weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    print(optimizer.param_groups[0]['lr'])
    # writer = SummaryWriter(os.path.join(data_path, 'tensorboard_log'))

    if args.gpu == 0:
        vis = Visdom()
    else:
        vis = None

    if do_load:
        model_name = 'model_15.pth.tar'
        state = load_model(model_name)
        if state:
            print('load ', model_name)
            state_dict = state['state_dict']
            load_epoch = state['epoch']
            best_predict = state['predict']

            model_state = model.state_dict()
            model_state.update(state_dict)

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v

            try:
                model.module.load_state_dict(state_dict)
            except:
                #model.load_state_dict(new_state_dict)
                model.load_state_dict(model_state)
        else:
            print('could find such a model')
    # model.apply(weight_init)
    if do_test:
        data_test = ITrackDataXgaze(dataPath=data_path, split='test')

        test_sampler = torch.utils.data.distributed.DistributedSampler(data_test, shuffle=False)
        test_batchsampler = torch.utils.data.sampler.BatchSampler(test_sampler, batch_size, drop_last=False)

        test_loader = data.DataLoader(
            data_test,
            batch_sampler=test_batchsampler,
            num_workers=workers, pin_memory=True
        )

        test(test_loader, model)

    if do_train:
        data_train = ITrackDataXgaze(dataPath=data_path)
        data_val = ITrackDataXgaze(dataPath=data_path, split='validate')

        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val, shuffle=True)
        train_batchsampler = torch.utils.data.sampler.BatchSampler(train_sampler, batch_size, drop_last=True)
        val_batchsampler = torch.utils.data.sampler.BatchSampler(val_sampler, batch_size, drop_last=True)

        train_loader = data.DataLoader(
            data_train,
            batch_sampler=train_batchsampler,
            num_workers=workers, pin_memory=True
        )
        val_loader = data.DataLoader(
            data_val,
            batch_sampler=val_batchsampler,
            num_workers=workers, pin_memory=True
        )
        if args.gpu == 0:
            logger.add(Path(data_path).joinpath('model_dir_temp2/model_val.log'),
                    filter=lambda x: '(validate)' in x['message'])
            logger.add(Path(data_path).joinpath('model_dir_temp2/model_train.log'),
                    filter=lambda x: '(train)' in x['message'])

        for epoch in range(0, load_epoch+1):
            adjust_lr(optimizer, epoch)
        for epoch in range(load_epoch+1, epochs):
            adjust_lr(optimizer, epoch)

            train(train_loader, model, optimizer, scheduler, epoch, vis,args.gpu)

            current_predict = validate(val_loader, model, epoch, vis, args.gpu)

            if current_predict < best_predict:
                is_best = True
                best_predict = current_predict
                print(best_predict, epoch)
            else:
                is_best = False

            if args.gpu == 0:
                save_model({'state_dict': model.state_dict(),
                            'epoch': epoch,
                            'predict': current_predict}, is_best)


def train(train_loader, model, optimizer, scheduler, epoch, vis, gpu):
    global count_train
    batch_time = AverageMeter()
    data_time = AverageMeter()
    compute_time = AverageMeter()
    update_loss_time = AverageMeter()
    losses = AverageMeter()
    ang_error = AverageMeter()
    print('begin to train')

    # switch to train mode
    model.train()

    end = time.time()

    title = 'loss_train_{}'.format(epoch)
    win = 'train_loss_{}'.format(epoch)

    for i, (imFace, imEyeL, imEyeR, gaze_direction) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        gaze_direction = gaze_direction.cuda()

        # imFace = torch.autograd.Variable(imFace, requires_grad=True)
        # imEyeL = torch.autograd.Variable(imEyeL, requires_grad=True)
        # imEyeR = torch.autograd.Variable(imEyeR, requires_grad=True)
        # gaze_direction = torch.autograd.Variable(gaze_direction, requires_grad=False)

        # compute output
        output = model(imFace, imEyeL, imEyeR)

        # loss = criterion(output, gaze_direction)
        loss = F.l1_loss(output, gaze_direction).cuda(gpu)
        print(output[0], gaze_direction[0])

        losses.update(loss.data.item(), imFace.size(0))
        print(imFace.size(0))

        error = angular_error(output.data.cpu().numpy(), gaze_direction.data.cpu().numpy())
        ang_error.update(error, imFace.size(0))

        #tensorboard show loss
        # writer.add_scalar('train_loss', losses.val, i)
        # writer.add_scalar('train_error', ang_error.val, i)
        if vis != None:
            vis.line(X=[count_train],Y=[losses.val],
                    win='loss_train',opts={'title':'entire_train_loss'},update='append')

            vis.line(X=[count_train], Y=[ang_error.val],
                    win='error_train', opts={'title': 'entire_train_error'}, update='append')

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        #print([x.grad for x in optimizer.param_groups[0]['params']])

        update_loss_time.update(time.time() - end - data_time.val - compute_time.val)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count_train = count_train + 1
        
        if args.gpu == 0:
            logger.info('Epoch (train): [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Error {ang_error.val:.4f} ({ang_error.avg:.4f})', epoch, i, len(train_loader),
                        loss=losses, ang_error=ang_error)

        # print('Epoch (train): [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #       'compute {compute_time.val:.3f} ({compute_time.avg:.3f})\t'
        #       'update_loss {update_loss_time.val:.3f} ({update_loss_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, compute_time=compute_time, update_loss_time=update_loss_time, loss=losses))

    scheduler.step()
    logger.info('\n-------------------------------------------------------------------\n'
                'Epoch (train):\t\t\t({0})\nLoss\t\t\t({loss.avg:.4f})\n'
                'Error\t\t\t({ang_error.avg:.4f})\nlr\t\t\t({lr_show})\n'
                '-------------------------------------------------------------------',
                epoch, loss=losses, ang_error=ang_error, lr_show=optimizer.param_groups[0]['lr'])


def validate(val_loader, model, epoch, vis, gpu):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ang_error = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    oIndex = 0
    for i, (imFace, imEyeL, imEyeR, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        gaze = gaze.cuda()

        # imFace = torch.autograd.Variable(imFace, requires_grad=False)
        # imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
        # imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)
        # gaze = torch.autograd.Variable(gaze, requires_grad=False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR)

        loss = F.l1_loss(output, gaze).cuda(gpu)
        print(output[0], gaze[0])

        error = angular_error(output.data.cpu().numpy(), gaze.data.cpu().numpy())
        ang_error.update(error, imFace.size(0))

        losses.update(loss.data.item(), imFace.size(0))

        # tensorboard show loss
        # writer.add_scalar('validate_loss', losses.val, i)
        # writer.add_scalar('validate_error', ang_error.val, i)
        if vis != None:
            vis.line(X=[count_test], Y=[losses.val], win='loss_validate',
                    opts={'title': 'entire_validate_loss'}, update='append')
            vis.line(X=[count_test], Y=[ang_error.val], win='error_validate',
                    opts={'title': 'entire_validate_error'}, update='append')

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count_test += 1

        if args.gpu == 0:
            logger.info('Epoch (validate): [{}][{}/{}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Error {ang_error.val:.4f} ({ang_error.avg:.4f})', epoch, i, len(val_loader),
                        loss=losses, ang_error=ang_error)

        # print('Epoch (val): [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #        epoch, i, len(val_loader), batch_time=batch_time,
        #        loss=losses))

    logger.info('\n-------------------------------------------------------------------\n'
                'Epoch (validate):\t\t\t({0})\nLoss\t\t\t({loss.avg:.4f})\n'
                'Error\t\t\t({ang_error.avg:.4f})\n'
                '-------------------------------------------------------------------',
                epoch, loss=losses, ang_error=ang_error)
    return losses.avg


@logger.catch
def test(test_loader, model):
    # switch to evaluate mode
    model.eval()
    save_index = 0
    test_num = len(test_loader.dataset)
    gaze_predict_all = np.zeros((test_num, 2))
    logger.add(Path(data_path).joinpath('test.log'), filter=lambda x: 'test' in x['message'])
    print('success to add logger')

    for i, (imFace, imEyeL, imEyeR, gaze) in enumerate(test_loader):
        # measure data loading time
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()

        imFace = torch.autograd.Variable(imFace, requires_grad=False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR)

        gaze_predict_all[save_index:save_index+output.shape[0], :] = output.data.cpu().numpy()
        save_index += output.shape[0]
        logger.info('[{}/{}] success to test this batch-size', i, len(test_loader))

    if save_index == test_num:
        print('the number match')

    np.savetxt(os.path.join(data_path, 'result.txt'), gaze_predict_all, delimiter=',')
    print('finish test')


def save_model(state, is_best, path=Path(data_path)):
    model_dir = path / 'model_dir_temp2'
    if not model_dir.is_dir():
        model_dir.mkdir()

    model_path = model_dir.joinpath('model_{}.pth.tar'.format(state['epoch']))
    if not model_path.is_file():
        torch.save(state, str(model_path))

    if is_best:
        best_model_path = model_dir.joinpath('best_model_{}.pth.tar'.format(state['epoch']))
        shutil.copyfile(str(model_path), str(best_model_path))


def load_model(model_name, path=Path(data_path)):
    model_dir = path / 'model_dir_temp2'
    if not model_dir.is_dir():
        print('invalidate model path')
        return None
    model_path = model_dir / model_name
    if not model_path.is_file():
        print('not such a model')
        return None
    state = torch.load(str(model_path), map_location='cpu')
    return state


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0

    def update(self, val, count=1):
        self.val = val
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch):
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def weight_init(m):
    # if isinstance(m, nn.Conv2d):
    #     init.xavier_uniform_(m.weight.data)
    #     init.constant_(m.bias.data,0.01)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.1)
        m.bias.data.zero_()


if __name__ == "__main__":
    main()
    print('finished')
