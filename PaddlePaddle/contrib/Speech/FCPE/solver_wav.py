# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import numpy as np
import paddle
import time
from savertools.saver import Saver
from savertools.saver import utils
from paddle.amp import auto_cast as autocast
from paddle.amp import GradScaler

from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm
import gc

USE_MIR = True

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
logger = Logger(
    [
        JSONStreamBackend(Verbosity.VERBOSE, "json_log.log"),
    ]
)

def test(args, model, loader_test, saver):
    print(' [*] testing...')
    logger.info(data=' [*] testing...')
    model.eval()

    # losses
    _rpa = _rca = _oa = _vfa = _vr = test_loss = 0.
    _num_a = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    count = 0
    start_time = time.time()
    with paddle.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data[2][0]
            print('--------')
            logger.info(data='--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))
            logger.info(data=dict(bidx=bidx,num_batches=num_batches,fn=fn))

            # unpack data
            # for k in data.keys():
            #    if not k.startswith('name'):
            #        data[k] = data[k].to(args.device)
            for k in range(len(data)):
                if k < 2:
                    data[k] = paddle.to_tensor(data[k])
            # print('>>', data[2][0])

            # forward
            st_time = time.time()
            f0 = model._layers.infer(mel=data[0])
            ed_time = time.time()

            if USE_MIR:
                _f0 = f0.squeeze().cpu().numpy()
                _df0 = data[1].squeeze().cpu().numpy()

                time_slice = np.array([i * args.mel.hop_size * 1000 / args.mel.sr for i in range(len(_df0))])
                ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, _df0, time_slice, _f0)

                rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
                rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
                oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
                vfa = voicing_false_alarm(ref_v, est_v)
                vr = voicing_recall(ref_v, est_v)

            # RTF
            run_time = ed_time - st_time
            song_time = f0.shape[1] * args.mel.hop_size / args.mel.sr
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            logger.info(data=dict(RTF=dict(rtf=rtf,run_time=run_time,song_time=song_time)))
            if USE_MIR:
                print('RPA: {}  | RCA: {} | OA: {} | VFA: {} | VR: {} |'.format(rpa, rca, oa, vfa, vr))
                logger.info(data=dict(RPA=rpa,RCA=rca,OA=oa,VFA=vfa,VR=vr))
            rtf_all.append(rtf)

            # loss
            for i in range(args.train.batch_size):
                loss = model._layers.train_and_loss(mel=data[0], gt_f0=data[1], loss_scale=args.loss.loss_scale)
                test_loss += loss.item()

            if USE_MIR:
                _rpa = _rpa + rpa
                _rca = _rca + rca
                _oa = _oa + oa
                _vfa = _vfa + vfa
                _vr = _vr + vr
                _num_a = _num_a + 1

            # log mel
            saver.log_spec(data[3][0], data[0], data[0])

            saver.log_f0(data[3][0], f0, data[1])
            saver.log_f0(data[3][0], f0, data[1], inuv=True)
            count += 1
        end_time = time.time()

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    if USE_MIR:
        _rpa /= _num_a

        _rca /= _num_a

        _oa /= _num_a

        _vfa /= _num_a

        _vr /= _num_a

    # check
    print(' [test_loss] test_loss:', test_loss)
    
    print(' Real Time Factor', np.mean(rtf_all))
    logger.info(data={"val.loss" : test_loss,
                      "val.Real_Time_Factor" : np.mean(rtf_all),
                      "val.ips" : count/(end_time - start_time)})
    return test_loss, _rpa, _rca, _oa, _vfa, _vr


def train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = paddle.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = paddle.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = paddle.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(args.train.epochs):
        train_one_epoch(loader_train, saver, optimizer, model, dtype, scaler, epoch, args, scheduler, num_batches,
                        loader_test)
        # 手动gc,防止内存泄漏
        gc.collect()


def train_one_epoch(loader_train, saver, optimizer, model, dtype, scaler, epoch, args, scheduler, num_batches,
                    loader_test):
    for batch_idx, data in enumerate(loader_train):
        train_one_step(batch_idx, data, saver, optimizer, model, dtype, scaler, epoch, args, scheduler, num_batches,
                       loader_test)


def train_one_step(batch_idx, data, saver, optimizer, model, dtype, scaler, epoch, args, scheduler, num_batches,
                   loader_test):
    saver.global_step_increment()
    optimizer.clear_grad()
    # unpack data
    for k in range(len(data)):
        if k < 2:
            data[k] = paddle.to_tensor(data[k])
    # print('>>', data[2][0])
    # forward
    if dtype == paddle.float32:
        loss = model._layers.train_and_loss(mel=data[0], gt_f0=data[1], loss_scale=args.loss.loss_scale)
    else:
        with autocast(enable = not args.device == 'cpu',
                      dtype=str(dtype).replace("paddle.",""),
                     custom_black_list=args.train.custom_black_list):
            loss = model._layers.train_and_loss(mel=data[0], gt_f0=data[1], loss_scale=args.loss.loss_scale)
    # handle nan loss
    if paddle.isnan(loss):
        # raise ValueError(' [x] nan loss ')
        print(' [x] nan loss ')
        loss = None
        return
    else:
        # backpropagate
        if dtype == paddle.float32:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    # log loss
    if saver.global_step % args.train.interval_log == 0:
        current_lr = scheduler.last_lr
        saver.log_info(
            'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                epoch,
                batch_idx,
                num_batches,
                args.env.expdir,
                args.train.interval_log / saver.get_interval_time(),
                current_lr,
                loss.item(),
                saver.get_total_time(),
                saver.global_step
            )
        )
        saver.log_value({
            'train/loss': loss.item()
        })

        saver.log_value({
            'train/lr': current_lr
        })

        logger.log(saver.global_step,{"rank":paddle.distributed.get_rank(),
                         "time":saver.get_total_time(),
                         "train.loss":loss.item(),
                         "train.epoch":epoch,
                         "train.batch_idx":batch_idx,
                         "train.batch_per_second":args.train.interval_log / saver.get_interval_time()})

    # validation
    if saver.global_step % args.train.interval_val == 0:
        optimizer_save = optimizer if args.train.save_opt else None

        # save latest
        saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}', config_dict=dict(args))
        last_val_step = saver.global_step - args.train.interval_val
        if last_val_step % args.train.interval_force_save != 0:
            saver.delete_model(postfix=f'{last_val_step}')

        # run testing set
        test_loss, rpa, rca, oa, vfa, vr = test(args, model, loader_test, saver)

        # log loss
        saver.log_info(
            ' --- <validation> --- \nloss: {:.3f}. '.format(
                test_loss,
            )
        )

        saver.log_value({
            'validation/loss': test_loss
        })
        if USE_MIR:
            saver.log_value({
                'validation/rpa': rpa
            })
            saver.log_value({
                'validation/rca': rca
            })
            saver.log_value({
                'validation/oa': oa
            })
            saver.log_value({
                'validation/vfa': vfa
            })
            saver.log_value({
                'validation/vr': vr
            })
        model.train()
