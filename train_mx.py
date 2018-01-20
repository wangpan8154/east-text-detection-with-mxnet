import mxnet as mx
import time
import os
import logging
import numpy as np
import json
import datetime
import glob
from east_symbol import east_symbol
from data_iter.dataset import TrainDataListDataset as dataset
from data_iter.dataloader import DataLoader as dataloader
from icdar_mx import save_train_data

def main():
    batch_per_gpu = 10
    num_gpus = 1
    num_workers = 5
    batch_size = int(batch_per_gpu * num_gpus)
    im_size = 512
    save_model_steps = 2
    update_data_steps = 1
    optimizer = 'adadelta'
    optimizer_params = {'rho': 0.99, 'wd': 0.00001}
    load_epoch = 0
    begin_epoch, num_epoch = load_epoch+1, 300
    train_eval_ratio = 10
    train_data_path = "/home/wangpan/Dataset/OCR/icdar2015/all_train_data/"
    model_prefix = "/home/wangpan/Workspace/OCR/EAST/model/text_det"
    result_path = "/home/wangpan/Workspace/OCR/EAST/output/"

    logger=set_logging(result_path)

    train_data, eval_data = train_eval_dataloader(train_data_path, batch_size, im_size, num_workers, train_eval_ratio)

    _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)
    train_net = east_symbol()
    sym = train_net.get_symbol(is_training=True)
    mod = mx.mod.Module(symbol=sym, context=[mx.gpu()], logger=logger,
                        data_names=('data',), label_names=('gt_cls','gt_geo','training_mask'))

    mod.bind(data_shapes=[('data',(batch_size, 3, im_size, im_size))],
             label_shapes=[('gt_cls',(batch_size, 1, im_size/4, im_size/4)),
                           ('gt_geo', (batch_size, 5, im_size/4, im_size/4)),
                           ('training_mask', (batch_size, 1, im_size/4, im_size/4))],
             for_training=True, force_rebind=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params,
                     allow_missing=True, force_init=False)
    mod.init_optimizer(kvstore='device', optimizer=optimizer,
                       optimizer_params=optimizer_params)

    for epoch in range(begin_epoch, begin_epoch+num_epoch):
        tic = time.time()
        nbatch = 0
        train_data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(train_data_iter)
        losses=[]
        while not end_of_batch:
            data_batch = next_data_batch
            mod.forward_backward(data_batch)
            mod.update()
            try:
                next_data_batch = next(train_data_iter)
                mod.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True
            get_loss= [loss.asnumpy()[0]*batch_size for loss in mod.get_outputs()]
            losses.append(get_loss)
            nbatch+=1
            print('Batch[{}] cls loss={:.4f} loc loss={:.4f}'.format(nbatch, get_loss[0], get_loss[1]))
        toc = time.time()
        mean_loss = np.array(losses).mean(0)
        info = 'Epoch[{}] Time cost={:.3f} Cls loss={:.4f} Loc loss={:.4f}'.format(epoch, (toc - tic), mean_loss[0], mean_loss[1])
        logger.info(info)

        eval_data_iter=iter(eval_data)
        next_data_batch = next(eval_data_iter)
        end_of_batch = False
        losses = []
        while not end_of_batch:
            data_batch = next_data_batch
            mod.forward(data_batch)
            try:
                next_data_batch = next(train_data_iter)
                mod.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True
            get_loss = [loss.asnumpy()[0] * batch_size for loss in mod.get_outputs()]
            losses.append(get_loss)
        mean_loss = np.array(losses).mean(0)
        info = 'Epoch[{}] evalidation Cls loss={:.4f} Loc loss={:.4f}'.format(epoch,mean_loss[0], mean_loss[1])
        logger.info(info)

        # sync aux params across devices
        arg_params, aux_params = mod.get_params()
        mod.set_params(arg_params, aux_params)

        if epoch%save_model_steps==0:
            mod.save_checkpoint(model_prefix, epoch)
        if (epoch+1-begin_epoch) % update_data_steps == 0:
            tic = time.time()
            save_train_data(input_data_path='/home/wangpan/Dataset/OCR/icdar2015/train/',
                            input_text_path='/home/wangpan/Dataset/OCR/icdar2015/traingt/',
                            ouput_data_path='/home/wangpan/Dataset/OCR/icdar2015/all_train_data/')
            train_data, eval_data = train_eval_dataloader(train_data_path, batch_size, im_size, num_workers, train_eval_ratio)
            print("gen data cost time={:.2f}s".format(time.time() - tic))
        train_data.reset()


def train_eval_dataloader(train_data_path, batch_size, im_size, num_workers, train_eval_ratio):
    samples_list = glob.glob(os.path.join(train_data_path, '*.npy'))
    ids = range(len(samples_list))
    np.random.shuffle(ids)
    train_list = [samples_list[i] for i in ids[len(ids) / train_eval_ratio:]]
    eval_list = [samples_list[i] for i in ids[:len(ids) / train_eval_ratio]]
    train_dataset = dataset(train_data_list=train_list)
    train_data = dataloader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            provide_data=[('data', (batch_size, 3, im_size, im_size))],
                            provide_label=[('gt_cls', (batch_size, 1, im_size / 4, im_size / 4)),
                                           ('gt_geo', (batch_size, 5, im_size / 4, im_size / 4)),
                                           ('training_mask', (batch_size, 1, im_size / 4, im_size / 4))])
    eval_dataset = dataset(train_data_list=eval_list)
    eval_data = dataloader(eval_dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           provide_data=[('data', (batch_size, 3, im_size, im_size))],
                           provide_label=[('gt_cls', (batch_size, 1, im_size / 4, im_size / 4)),
                                          ('gt_geo', (batch_size, 5, im_size / 4, im_size / 4)),
                                          ('training_mask', (batch_size, 1, im_size / 4, im_size / 4))])
    return train_data, eval_data

def set_logging(output_path):
    now = datetime.datetime.now()
    nowtime = now.strftime("%m-%d-%H:%M:%S")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path,nowtime+'.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

if __name__ == '__main__':
    main()
