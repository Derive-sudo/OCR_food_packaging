# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 本项目微调专用训练入口：在官方 tools/train.py 基础上，增加「冻结 Backbone」支持。
# 通过配置里的 Global.freeze_backbone=true 触发（finetune_rec.py 生成配置时写入）。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# 定位 PaddleOCR 源码目录（本文件位于 model_train/ 下，与 PaddleOCR/ 同级）
__dir__ = os.path.dirname(os.path.abspath(__file__))
PADDLEOCR_DIR = os.path.join(__dir__, "PaddleOCR")
sys.path.insert(0, PADDLEOCR_DIR)
sys.path.insert(0, os.path.join(PADDLEOCR_DIR, "tools"))

import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
import tools.program as program

dist.get_world_size()


def freeze_backbone(model):
    """把 Backbone 参数置为不可训练（stop_gradient），优化器只会更新 Head。"""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.stop_gradient = True
            frozen += 1
    return frozen


def main(config, device, logger, vdl_writer, seed):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    set_signal_handlers()
    train_dataloader = build_dataloader(config, 'Train', device, logger, seed)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger,
                                            seed)
    else:
        valid_dataloader = None
    step_pre_epoch = len(train_dataloader)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    if config['PostProcess'][
                            'name'] == 'DistillationNRTRLabelDecode':
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    # update SARLoss params
                    if list(config['Loss']['loss_config_list'][-1].keys())[
                            0] == 'DistillationSARLoss':
                        config['Loss']['loss_config_list'][-1][
                            'DistillationSARLoss'][
                                'ignore_index'] = char_num + 1
                        out_channels_list['SARLabelDecode'] = char_num + 2
                    elif any('DistillationNRTRLoss' in d
                             for d in config['Loss']['loss_config_list']):
                        out_channels_list['NRTRLabelDecode'] = char_num + 3

                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            if config['PostProcess']['name'] == 'NRTRLabelDecode':
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list['CTCLabelDecode'] = char_num
            # update SARLoss params
            if list(config['Loss']['loss_config_list'][1].keys())[
                    0] == 'SARLoss':
                if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                    config['Loss']['loss_config_list'][1]['SARLoss'] = {
                        'ignore_index': char_num + 1
                    }
                else:
                    config['Loss']['loss_config_list'][1]['SARLoss'][
                        'ignore_index'] = char_num + 1
                out_channels_list['SARLabelDecode'] = char_num + 2
            elif list(config['Loss']['loss_config_list'][1].keys())[
                    0] == 'NRTRLoss':
                out_channels_list['NRTRLabelDecode'] = char_num + 3
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

        if config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
            config['Loss']['ignore_index'] = char_num - 1

    model = build_model(config['Architecture'])

    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info('convert_sync_batchnorm')

    model = apply_to_static(model, config, logger)

    # ---- 本项目新增：冻结 Backbone（必须在 build_optimizer 之前，因为优化器只取 trainable 参数）----
    if global_config.get('freeze_backbone', False):
        n = freeze_backbone(model)
        logger.info('freeze backbone: {} params set stop_gradient'.format(n))

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model)

    # build metric
    eval_class = build_metric(config['Metric'])

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", 'O2')
    amp_dtype = config["Global"].get("amp_dtype", 'float16')
    amp_custom_black_list = config['Global'].get('amp_custom_black_list', [])
    amp_custom_white_list = config['Global'].get('amp_custom_white_list', [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update({
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_gemm_use_half_precision_compute_type': 0,
            })
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False)
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype)
    else:
        scaler = None

    # load pretrain model
    pre_best_model_dict = load_model(config, model, optimizer,
                                     config['Architecture']["model_type"])

    if config['Global']['distributed']:
        model = paddle.DataParallel(model)
    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, step_pre_epoch,
                  vdl_writer, scaler, amp_level, amp_custom_black_list,
                  amp_custom_white_list, amp_dtype)


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer, seed)
