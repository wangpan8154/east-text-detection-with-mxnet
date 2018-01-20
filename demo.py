import cv2
import time
import os
import numpy as np
import mxnet as mx

import locality_aware_nms as nms_locality
import lanms
from icdar_mx import restore_rectangle
from east_symbol import east_symbol as eastnet


class get_params(object):
    def __init__(self, test_data_path='./demo_images/',
                 checkpoint_path='./east_icdar2015_resnet_v1_50_rbox/',
                 output_dir="./output/",
                 model_prefix='./model/text_det',
                 gpu_list='0',
                 model_epoch=26,
                 no_write_images=False):
        self.test_data_path = test_data_path
        self.gpu_list = gpu_list
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.no_write_images = no_write_images
        self.model_prefix = model_prefix
        self.model_epoch = model_epoch


FLAGS = get_params(model_epoch=26)


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def process_image(im, max_side_len=2304):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    # means = [123.68, 116.78, 103.94]
    # means = [0, 0, 0]

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    im = im[:, :, ::-1]
    # for i in range(3):
    #     im[:,:,i]=im[:,:,i]-means[i]
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im = im[np.newaxis, :]
    return im, (ratio_h, ratio_w)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def extract_model(variables_to_restore, f_variables_to_restore):
    def __name_in_checkpoint(i, var):
        if 'resnet_v1_50' in var.op.name:
            if i <= 4:
                if "weights" in var.op.name:
                    return 'conv0_weight'
                if "beta" in var.op.name:
                    return 'bn0_beta'
                if "gamma" in var.op.name:
                    return 'bn0_gamma'
                if "moving_mean" in var.op.name:
                    return 'bn0_moving_mean'
                if "moving_variance" in var.op.name:
                    return 'bn0_moving_var'
            elif i <= 269:
                nams = var.op.name.split('/')
                pre_name = nams[1].replace("block", "stage") + nams[2].replace("unit_", "_unit")
                if "conv" in var.op.name:
                    if "weights" in var.op.name:
                        return pre_name + "_" + nams[4] + "_weight"
                    if "beta" in var.op.name:
                        return pre_name + "_bn" + nams[4][-1] + "_beta"
                    if "gamma" in var.op.name:
                        return pre_name + "_bn" + nams[4][-1] + '_gamma'
                    if "moving_mean" in var.op.name:
                        return pre_name + "_bn" + nams[4][-1] + '_moving_mean'
                    if "moving_variance" in var.op.name:
                        return pre_name + "_bn" + nams[4][-1] + '_moving_var'
                elif "shortcut" in var.op.name:
                    if "weights" in var.op.name:
                        return pre_name + "_" + "sc_conv_weight"
                    if "beta" in var.op.name:
                        return pre_name + "_sc_bn_beta"
                    if "gamma" in var.op.name:
                        return pre_name + "_sc_bn_gamma"
                    if "moving_mean" in var.op.name:
                        return pre_name + "_sc_bn_moving_mean"
                    if "moving_variance" in var.op.name:
                        return pre_name + "_sc_bn_moving_var"
        else:
            nams = var.op.name.split('/')
            if "weights" in var.op.name:
                if '_' in nams[1]:
                    return "convolution{}_weight".format(nams[1].split('_')[-1])
                else:
                    return "convolution0_weight"

            if "biases" in var.op.name:
                if '_' in nams[1]:
                    return "convolution{}_bias".format(nams[1].split('_')[-1])
                else:
                    return "convolution0_bias"

            if "beta" in var.op.name:
                if '_' in nams[1]:
                    return "batchnorm{}_beta".format(nams[1].split('_')[-1])
                else:
                    return "batchnorm0_beta"

            if "gamma" in var.op.name:
                if '_' in nams[1]:
                    return "batchnorm{}_gamma".format(nams[1].split('_')[-1])
                else:
                    return "batchnorm0_gamma"

            if "moving_mean" in var.op.name:
                if '_' in nams[1]:
                    return "batchnorm{}_moving_mean".format(nams[1].split('_')[-1])
                else:
                    return "batchnorm0_moving_mean"

            if "moving_variance" in var.op.name:
                if '_' in nams[1]:
                    return "batchnorm{}_moving_var".format(nams[1].split('_')[-1])
                else:
                    return "batchnorm0_moving_var"

    arg_param = {}
    aux_param = {}
    to_restore = {__name_in_checkpoint(i, var): variables_to_restore[i] for i, var in enumerate(f_variables_to_restore)}
    for l in to_restore:
        if l:
            if 'moving' in l:
                aux_param.update({l: mx.nd.array(to_restore[l], ctx=mx.gpu())})
            else:
                d = to_restore[l]
                if 'weight' in l:
                    d = np.transpose(d, (3, 2, 1, 0))
                arg_param.update({l: mx.nd.array(d, ctx=mx.gpu())})

    resnet = resnet50()
    sym = resnet.get_symbol()
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg_param, aux_params=aux_param, allow_extra=True, allow_missing=True)
    mod.save_checkpoint(prefix='/home/wangpan/Workspace/OCR/EAST/new/text_det', epoch=0)
    return mod


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, 0, :, :]
        geo_map = np.transpose(geo_map[0, :, :, ], (1, 2, 0))
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer
    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer


def main():
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=FLAGS.model_prefix, epoch=FLAGS.model_epoch)
    sym = eastnet().get_symbol()
    mod = mx.mod.Module(symbol=sym, context=[mx.gpu()], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 640, 1152))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    im_fn_list = get_images()
    for im_fn in im_fn_list:
        im = cv2.imread(im_fn)
        start_time = time.time()
        im = cv2.resize(im, (1152, 640))
        im_resized, (ratio_h, ratio_w) = process_image(im)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        # mx_mod = extract_model(variables_to_restore, f_variables_to_restore)
        data_batch = mx.io.DataBatch(data=[mx.nd.array(im_resized)], label=[], pad=0, index=0,
                                     provide_data=[('data', im_resized.shape)], provide_label=[None])
        mod.forward(data_batch=data_batch)
        score = mod.get_outputs()[0].asnumpy()
        geometry = mod.get_outputs()[1].asnumpy()

        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        # save to file
        if boxes is not None:
            res_file = os.path.join(
                FLAGS.output_dir,
                '{}.txt'.format(
                    os.path.basename(im_fn).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(im[:, :, :], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                                  thickness=2)
        if not FLAGS.no_write_images:
            img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
            cv2.imwrite(img_path, im[:, :, :])


if __name__ == '__main__':
    main()
