import mxnet as mx
import numpy as np


class east_symbol(object):
    def __init__(self):
        self.bn_eps = 1e-5
        self.bn_mom = 0.997

    def resnet50_v1_symbol(self, x):
        ## stage1
        a_stride = (1, 1)
        name = "stage1_unit1_"
        num_filter = 256
        bypass = mx.sym.Convolution(data=x, num_filter=num_filter, kernel=(1, 1), stride=a_stride, pad=(0, 0),
                                    no_bias=True, name=name + "sc_conv")
        bypass = mx.sym.BatchNorm(data=bypass, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                  momentum=self.bn_mom, name=name + "sc_bn")
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage1_unit2_"
        num_filter = 256
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (2, 2)
        name = "stage1_unit3_"
        num_filter = 256
        bypass = mx.sym.Pooling(x, kernel=(1, 1), pool_type='max', stride=(2, 2))
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride, out_name="relu_pool3")

        ## stage2
        a_stride = (1, 1)
        name = "stage2_unit1_"
        num_filter = 512
        bypass = mx.sym.Convolution(data=x, num_filter=num_filter, kernel=(1, 1), stride=a_stride, pad=(0, 0),
                                    no_bias=True,
                                    name=name + "sc_conv")
        bypass = mx.sym.BatchNorm(data=bypass, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                  momentum=self.bn_mom, name=name + "sc_bn")
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage2_unit2_"
        num_filter = 512
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage2_unit3_"
        num_filter = 512
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (2, 2)
        name = "stage2_unit4_"
        num_filter = 512
        bypass = mx.sym.Pooling(x, kernel=(1, 1), pool_type='max', stride=(2, 2))
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride, out_name="relu_pool4")

        ## stage3
        a_stride = (1, 1)
        name = "stage3_unit1_"
        num_filter = 1024
        bypass = mx.sym.Convolution(data=x, num_filter=num_filter, kernel=(1, 1), stride=a_stride, pad=(0, 0),
                                    no_bias=True,
                                    name=name + "sc_conv")
        bypass = mx.sym.BatchNorm(data=bypass, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                  momentum=self.bn_mom, name=name + "sc_bn")
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage3_unit2_"
        num_filter = 1024
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage3_unit3_"
        num_filter = 1024
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage3_unit4_"
        num_filter = 1024
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage3_unit5_"
        num_filter = 1024
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (2, 2)
        name = "stage3_unit6_"
        num_filter = 1024
        bypass = mx.sym.Pooling(x, kernel=(1, 1), pool_type='max', stride=(2, 2))
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        ## stage4
        a_stride = (1, 1)
        name = "stage4_unit1_"
        num_filter = 2048
        bypass = mx.sym.Convolution(data=x, num_filter=num_filter, kernel=(1, 1), stride=a_stride, pad=(0, 0),
                                    no_bias=True, name=name + "sc_conv")
        bypass = mx.sym.BatchNorm(data=bypass, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                  momentum=self.bn_mom, name=name + "sc_bn")
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage4_unit2_"
        num_filter = 2048
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride)

        a_stride = (1, 1)
        name = "stage4_unit3_"
        num_filter = 2048
        bypass = x
        x = self.__res_base_conv(x, bypass, num_filter, name, a_stride, out_name="relu_pool5")
        return x

    def __res_base_conv(self, x, bypass, num_filter, name, a_stride, out_name=None):
        if out_name:
            relu_name = out_name
        else:
            relu_name = name + "relu3"
        x = mx.sym.Convolution(data=x, num_filter=num_filter // 4, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, name=name + "conv1")
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, use_global_stats=False, eps=self.bn_eps, momentum=self.bn_mom,
                             name=name + "bn1")
        x = mx.sym.Activation(data=x, act_type='relu', name=name + "relu1")

        x = mx.sym.Convolution(data=x, num_filter=num_filter // 4, kernel=(3, 3), stride=a_stride, pad=(1, 1),
                               no_bias=True, name=name + "conv2")
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, use_global_stats=False, eps=self.bn_eps, momentum=self.bn_mom,
                             name=name + "bn2")
        x = mx.sym.Activation(data=x, act_type='relu', name=name + "relu2")

        x = mx.sym.Convolution(data=x, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               name=name + "conv3")
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, use_global_stats=False, eps=self.bn_eps, momentum=self.bn_mom,
                             name=name + "bn3")
        x = mx.sym.Activation(data=x + bypass, act_type='relu', name=relu_name)
        return x

    def mobilenet_v1_symbol(self):
        pass

    def get_loss(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):

        intersection = mx.sym.sum(y_true_cls * y_pred_cls * training_mask)
        union = mx.sym.sum(y_true_cls * training_mask) + mx.sym.sum(y_pred_cls * training_mask) + self.bn_eps
        classification_loss = 0.01 * (1. - (2 * intersection / union))
        cls_loss = mx.sym.MakeLoss(classification_loss)

        # scale classification loss to match the iou loss part

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = mx.sym.split(data=y_true_geo, num_outputs=5, axis=1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = mx.sym.split(data=y_pred_geo, num_outputs=5, axis=1)

        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

        w_union = mx.sym.broadcast_minimum(d2_gt, d2_pred) + mx.sym.broadcast_minimum(d4_gt, d4_pred)
        h_union = mx.sym.broadcast_minimum(d1_gt, d1_pred) + mx.sym.broadcast_minimum(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -mx.sym.log(data=(area_intersect + 1.0) / (area_union + 1.0))

        L_theta = 1 - mx.sym.cos(data=theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        location_loss = mx.sym.mean(data=L_g * y_true_cls * training_mask)
        loc_loss = mx.sym.MakeLoss(location_loss)
        # a, b, c=total_loss.infer_shape(gt_geo=(1, 5, 15, 15),pr_geo=(1, 5, 15, 15),
        #                          gt_cls=(1, 1, 15, 15),pr_cls=(1, 1, 15, 15),
        #                          training_mask=(1,1,15,15))

        return cls_loss, loc_loss  # , loc_loss, thd_loss, cls_loss

    def get_symbol(self, is_training=False, means=[124, 117, 104]):
        # means=[123.68, 116.78, 103.94]
        x = mx.sym.Variable(name='data')
        x0, x1, x2 = mx.sym.split(data=x, num_outputs=3, axis=1)
        x = mx.sym.Concat(*[x0 - means[0], x1 - means[1], x2 - means[2]], dim=1)
        # a,b,c=x.infer_shape(data=(10,3,704,1280))

        x = mx.sym.Convolution(data=x, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True,
                               name='conv0')
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, use_global_stats=False, eps=self.bn_eps, momentum=self.bn_mom,
                             name='bn0')
        x = mx.sym.Activation(data=x, act_type='relu', name='relu0')

        x = mx.sym.Pooling(data=x, kernel=(3, 3), pool_type='max', stride=(2, 2), pooling_convention='full',
                           name='pool2')
        x = self.resnet50_v1_symbol(x)

        res_sym = x.get_internals()
        f = [res_sym["relu_pool5_output"], res_sym["relu_pool4_output"],
             res_sym["relu_pool3_output"], res_sym["pool2_output"]]
        # v=[i.infer_shape(data=(10,3,704,1280))[1] for i in f]
        g = [None, None, None, None]
        h = [None, None, None, None]
        num_outputs = [None, 128, 64, 32]
        umsample_size = [2048, 128, 64]
        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                cur_data = mx.sym.Concat(*[g[i - 1], f[i]], dim=1)
                c1_1 = mx.sym.Convolution(data=cur_data, num_filter=num_outputs[i], kernel=(1, 1), no_bias=True)
                c1_1 = mx.sym.BatchNorm(data=c1_1, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                        momentum=self.bn_mom)
                c1_1 = mx.sym.Convolution(data=c1_1, num_filter=num_outputs[i], kernel=(3, 3), pad=(1, 1), no_bias=True)
                h[i] = mx.sym.BatchNorm(data=c1_1, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                        momentum=self.bn_mom)
                # a, b, c = h[i].infer_shape(data=(10, 3, 704,1280))
            if i <= 2:
                # a, b, c = h[i].infer_shape(data=(10, 3, 704,1280))
                g[i] = mx.sym.UpSampling(h[i], scale=2, sample_type='bilinear',
                                         num_filter=umsample_size[i])  # ,attr={'lr_mult':0}
            else:
                g[i] = mx.sym.Convolution(data=h[i], num_filter=num_outputs[i], kernel=(3, 3), pad=(1, 1), no_bias=True)
                g[i] = mx.sym.BatchNorm(data=g[i], fix_gamma=False, use_global_stats=False, eps=self.bn_eps,
                                        momentum=self.bn_mom)

        y_pred_cls = mx.sym.Convolution(data=g[3], num_filter=1, kernel=(1, 1))
        y_pred_cls = mx.sym.Activation(data=y_pred_cls, act_type='sigmoid')
        # 4 channel of axis aligned bbox and 1 channel rotation angle
        geo_map = mx.sym.Convolution(data=g[3], num_filter=4, kernel=(1, 1))
        geo_map = mx.sym.Activation(data=geo_map, act_type='sigmoid') * 512
        angle_map = mx.sym.Convolution(data=g[3], num_filter=1, kernel=(1, 1))
        angle_map = (mx.sym.Activation(data=angle_map, act_type='sigmoid') - 0.5) * np.pi / 2  # [-45, 45]
        y_pred_geo = mx.sym.Concat(*[geo_map, angle_map], dim=1)
        # a,b,c=y_pred_geo.infer_shape(data=(10,3,512,512))
        ## loss
        if is_training:
            y_true_cls = mx.sym.Variable(name='gt_cls')
            y_true_geo = mx.sym.Variable(name='gt_geo')
            training_mask = mx.sym.Variable(name='training_mask')
            cls_loss, loc_loss = self.get_loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask)
            result = mx.sym.Group([cls_loss, loc_loss])
        else:
            result = mx.sym.Group([y_pred_cls, y_pred_geo])
        # a,b,c=result.infer_shape(data=(10,3,704,1280))
        return result

# model = east_symbol()

# sy = model.get_symbol(is_training=False)
# # a=sy.get_internals()
# print a

# pass
