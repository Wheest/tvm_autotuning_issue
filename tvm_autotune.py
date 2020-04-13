#!/usr/bin/env python
# coding: utf-8

import os, sys
from shutil import copyfile
import numpy as np
import argparse
import tvm
from tvm import autotvm
from tvm import te
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
import pickle
import onnx
import tvm.relay as relay
from tvm.contrib.util import tempdir
import logging
sys.path.append('..')



def get_network(model_name, input_data, input_names, batch_size):
    """Get the symbol definition and random weight of a network
    Hacked for example
    """
    if model_name == 'WRN-40-2':
        out_shape = (batch_size, 10)
    elif model_name == 'resnet34':
        out_shape = (batch_size, 1000)
    shape_dict = {input_names[0]: input_data.shape}
    model = onnx.load(f'{model_name}.onnx')
    mod, params = relay.frontend.from_onnx(model, shape_dict)

    return mod, params


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)



def tune_and_evaluate(tuning_opt, model_name, device_info, input_data, input_names):
    # extract workloads from relay program
    print("Extract tasks...")
    dtype = "float32"
    target = tvm.target.create(device_info['target'])

    data_shape = input_data.shape
    mod, params = get_network(model_name, input_data, input_names, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    # pick best records to a cache file

    # compile kernels with history best records
    with autotvm.apply_history_best(tuning_option['log_filename']):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device
        if device_info['remote']:
            # export library
            tmp = tempdir()
            if use_android:
                from tvm.contrib import ndk
                filename = "net.so"
                lib.export_library(tmp.relpath(filename), ndk.create_shared)
            else:
                filename = "net.tar"
                lib.export_library(tmp.relpath(filename))

            # upload module to device
            print("Upload...")
            remote = autotvm.measure.request_remote(device_info['device_key'],
                                                    device_info['rpc_address'],
                                                    device_info['rpc_port'],
                                                    timeout=10000)
            remote.upload(tmp.relpath(filename))
            rlib = remote.load_module(filename)

            ctx = remote.context(str(target), 0)
            module = runtime.create(graph, rlib, ctx)
        else:
            ctx = tvm.cpu()
            module = runtime.create(graph, lib, ctx)

        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))

        module.set_input(input_names[0], data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tvm autotuning')
    parser.add_argument('--target_type', choices=['local', 'remote'], default='local', help='will tvm run models locally, or on a remote device')
    parser.add_argument('--device_key', default='hikey', help='key of target device')
    parser.add_argument('--host_port', default=9190, type=int, help='Port that RPC server is running on')
    parser.add_argument('--target_string', default='None', type=str, help='tvm target string, e.g. llvm')
    parser.add_argument('--output_path', default='/tmp/', type=str,
                        help='Output directory to save tuned results')
    parser.add_argument('--trials', default=2000, type=int, help='Number of iterations per task')
    parser.add_argument('--model', default='WRN-40-2', const='WRN-40-2', nargs='?',
                        choices=['WRN-40-2', 'resnet34'], help='Model to run')
    args = parser.parse_args()
    use_android = False

    logger = logging.getLogger('scope.name')
    file_log_handler = logging.FileHandler('logfile.log')
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    # nice output format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    device_info = {
        'device_key': args.device_key,
        'rpc_address': "0.0.0.0",
        'rpc_port': args.host_port,
        'target' : args.target_string
    }

    tuning_option = {
        'tuner': 'xgb_knob',
        'n_trial': args.trials,
        'early_stopping': 1000,
    }
    # local
    if args.target_type == "local":
        device_info['remote'] = False

        tuning_option['measure_option'] = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=1, repeat=1,
                                       min_repeat_ms=1000))
    elif args.target_type == "remote":
        device_info['remote'] = True
        tuning_option['measure_option'] = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func='ndk' if use_android else 'default'),
            runner=autotvm.RPCRunner(device_info['device_key'],
                                     host=device_info['rpc_address'],
                                     port=device_info['rpc_port'],
                                     number=5, timeout=10,))

    print(device_info)

    log_file = os.path.join(args.output_path, args.model + '_tvm.log')
    if os.path.exists(log_file):
        print('deleting file')
        os.remove(log_file)
    os.mknod(log_file)
    tuning_option['log_filename'] = log_file
    print(log_file)

    batch_size = 1
    if args.model == 'WRN-40-2':
        in_shape = (batch_size, 3, 32, 32)
    elif model_name == 'resnet34':
        in_shape = (batch_size, 3, 224, 224)

    input_data = np.random.randint(-1, high=1, size=in_shape)
    input_names = ['input_tens']
    tune_and_evaluate(tuning_option, args.model, device_info,
                      input_data, input_names)
