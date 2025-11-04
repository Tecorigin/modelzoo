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

"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from io import open
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tqdm import tqdm, trange
from utils import data_loader

import modeling
from optimization import BertAdam, BertSumAdam, warmup_linear
from utils.utils import is_main_process, format_step
import dllogger, time

from utils.rouge_utils import test_rouge

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch_sdaa
except:
    from apex import amp
    print("no device sdaa")


# def warmup_linear(x, warmup=0.002):
#     if x < warmup:
#         return x/warmup
#     return max((x - 1. )/ (warmup - 1.), 0.)

json_logger = Logger(
[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})


try:
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    pass

class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """
    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )


def calculate_rouge(args, model, test_iter, cal_lead=False, cal_oracle=False):
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s))>0:
                return True
        return False

    if (not cal_lead and not cal_oracle):
        model.eval()

    args.recall_eval=False
    args.block_trigram=True


    can_path = os.path.join(args.output_dir, 'candidate.log')
    gold_path = os.path.join(args.output_dir, 'gold.log')
    with open(can_path, 'w') as save_pred:
        with open(gold_path, 'w') as save_gold:
            with torch.no_grad():
                for batch in test_iter:
                    src = batch.src
                    labels = batch.labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask
                    mask_cls = batch.mask_cls


                    gold = []
                    pred = []

                    if (cal_lead):
                        selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                    elif (cal_oracle):
                        selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                        range(batch.batch_size)]
                    else:
                        sent_scores, mask = model(src, segs, clss, mask, mask_cls)

                        sent_scores = sent_scores + mask.float()
                        sent_scores = sent_scores.cpu().data.numpy()
                        selected_ids = np.argsort(-sent_scores, 1)
                    # selected_ids = np.sort(selected_ids,1)
                    for i, idx in enumerate(selected_ids):
                        _pred = []
                        if(len(batch.src_str[i])==0):
                            continue
                        for j in selected_ids[i][:len(batch.src_str[i])]:
                            if(j>=len( batch.src_str[i])):
                                continue
                            candidate = batch.src_str[i][j].strip()
                            if(args.block_trigram):
                                if(not _block_tri(candidate,_pred)):
                                    _pred.append(candidate)
                            else:
                                _pred.append(candidate)

                            if ((not cal_oracle) and (not args.recall_eval) and len(_pred) == 3):
                                break

                        _pred = '<q>'.join(_pred)
                        if(args.recall_eval):
                            _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                        pred.append(_pred)
                        gold.append(batch.tgt_str[i])

                    for i in range(len(gold)):
                        save_gold.write(gold[i].strip()+'\n')
                    for i in range(len(pred)):
                        save_pred.write(pred[i].strip()+'\n')
    # if(step!=-1 and args.report_rouge):
    rouges = test_rouge(os.path.join(args.output_dir, "temp"), can_path, gold_path)
    logger.info('Rouges \n%s' % (rouge_results_to_str(rouges)))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="CNN/DM data for predictions. E.g., cnndm.test.0.pt")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--device",
                        default='sdaa', type=str, choices=['cpu', 'cuda', 'sdaa'],
                        help="which device to use, sdaa default")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local-rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--log_freq',
                        type=int, default=50,
                        help='frequency of logging loss.')
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--disable-progress-bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                        'the specified file.')
    
    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp

    if args.local_rank == -1:
        if args.device == "sdaa":
            device = torch.device(args.device if torch.sdaa.is_available() else 'cpu')
        elif args.device == "cuda":
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        # n_gpu = torch.sdaa.device_count()
        n_gpu = 1
    else:
        if args.device == "sdaa":
            torch.sdaa.set_device(args.local_rank)
            device = torch.device("sdaa", args.local_rank)
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='tccl')
        elif args.device == "cuda":
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        else:
            raise Exception("CPU do not support data parallel !")
        n_gpu = 1

    if is_main_process():
        Path(os.path.dirname(args.json_summary)).mkdir(parents=True, exist_ok=True)
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])
    
    dllogger.metadata("e2e_train_time", {"unit": "s"})
    dllogger.metadata("training_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("final_loss", {"unit": None})
    dllogger.metadata("e2e_inference_time", {"unit": "s"})
    dllogger.metadata("inference_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("exact_match", {"unit": None})
    dllogger.metadata("F1", {"unit": None})

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                                device, n_gpu, bool(args.local_rank != -1), args.fp16))

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    if n_gpu > 0:
        if args.device == "sdaa":
            torch.sdaa.manual_seed_all(args.seed)
        else:
            torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and os.listdir(args.output_dir)!=['logfile.txt']:
        print("WARNING: Output directory {} already exists and is not empty.".format(args.output_dir), os.listdir(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    num_train_optimization_steps = args.max_steps

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForTextSummarization(config)

    # model = modeling.BertForQuestionAnswering.from_pretrained(args.bert_model,
                # cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
    dllogger.log(step="PARAMETER", data={"loading_checkpoint": True})
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
    checkpoint = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
    model.load_state_dict(checkpoint, strict=False)
    dllogger.log(step="PARAMETER", data={"loaded_checkpoint": True})
    model.to(device)
    num_weights = sum([p.numel() for p in model.parameters() if p.requires_grad])
    dllogger.log(step="PARAMETER", data={"model_weights_num":num_weights})

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # if torch.distributed.get_rank()==0:
    #     print(optimizer_grouped_parameters)
    if args.do_train:
        if args.fp16:
            if args.device == "cuda":
                try:
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
                optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    bias_correction=False)

                if args.loss_scale == 0:
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                        loss_scale="dynamic")
                else:
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale=args.loss_scale)
            else:
                optimizer = BertSumAdam(param_optimizer, args.learning_rate,
                    decay_method='noam',
                    warmup_steps=num_train_optimization_steps*args.warmup_proportion)
                # optimizer.set_parameters(param_optimizer)
                from torch_sdaa.amp import GradScaler
                grad_scaler = GradScaler(init_scale=float(args.loss_scale), enabled=True)
            # if args.do_train:
            #     scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=num_train_optimization_steps)

        else:
            optimizer = BertSumAdam(param_optimizer, args.learning_rate,
                    decay_method='noam',
                    warmup_steps=num_train_optimization_steps*args.warmup_proportion)


    if args.local_rank != -1:
        if args.device == "cuda":
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif args.device == "sdaa":
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    if args.do_train:

        model.train()
        if args.device == "cuda":
            gradClipper = GradientClipper(max_grad_norm=1.0)
        final_loss = None
        def train_iter_fct():
            return data_loader.Dataloader(args, data_loader.load_dataset(args.train_file, 'train', shuffle=False), args.train_batch_size, device,
                                                 shuffle=True, is_test=False)
        train_start = time.time()
        
        train_iter = train_iter_fct()
        epoch = 0
        while global_step < args.max_steps:
            # TODO: 现在的batch为bertSUM的Batch对象，需要让input_ids, input_mask, segment_ids与Batch对齐
            # 整个代码的核在start_logits, end_logits = model(input_ids, segment_ids, input_mask)，之需要让输入的三个参数与Batch对齐即可
            
            for step, batch in enumerate(train_iter):
                before_to_device = time.time()
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
                
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                before_forward = time.time()
                if args.device == "sdaa" and args.fp16:
                    with torch_sdaa.amp.autocast():
                        sent_scores, mask_cls = model(src, segs, clss, mask, mask_cls)
                else:
                    sent_scores, mask_cls = model(src, segs, clss, mask, mask_cls)

                loss_fct = torch.nn.BCELoss(reduction='none')
                loss = loss_fct(sent_scores, labels.float())
                loss = (loss*mask_cls.float()).sum()

                forward_time = time.time() - before_forward
                
                loss_to_backward = loss/loss.numel()
                if args.gradient_accumulation_steps > 1:
                    loss = loss_to_backward / args.gradient_accumulation_steps
                
                before_backward = time.time()
                if args.fp16:
                    if args.device == "sdaa":
                        grad_scaler.scale(loss_to_backward).backward()
                    else:
                        with amp.scale_loss(loss_to_backward, optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss_to_backward.backward()

                backward_time = time.time() - before_backward

                before_optim = time.time()
                # gradient clipping
                if args.device == "cuda":
                    gradClipper.step(amp.master_params(optimizer))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.device == "sdaa" and args.fp16:
                        # scheduler.step()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    elif args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        # scheduler.step()
                        optimizer.step()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                optim_time = time.time() - before_optim
                # if is_main_process():
                #     if args.fp16:
                #         print(f"{grad_scaler._scale.item()=}")
                #     print(f"{args.local_rank=}, {global_step=}, {src.shape=}, {loss.item()=}")
                    # print("optimizer end")
                final_loss = loss_to_backward.item()
                batch_time=time.time() - before_to_device
                json_logger.log(
                    step = (epoch, global_step),
                    data = {
                            "rank":args.local_rank,
                            "train.loss":final_loss,
                            "train.ips":args.train_batch_size/batch_time,
                            "data.shape":src.shape,
                            "train.lr":optimizer.learning_rate,
                            "train.data_time":-1,
                            "train.compute_time": forward_time+backward_time+optim_time,
                            "train.fp_time":forward_time,
                            "train.bp_time":backward_time,
                            "train.grad_time":optim_time,
                            },
                    verbosity=Verbosity.DEFAULT,
                )
                if global_step % args.log_freq == 0:
                    dllogger.log(step=(epoch, global_step,), data={"step_loss": final_loss,
                                                                    "learning_rate": optimizer.param_groups[0]['lr']})
                
                # if global_step % 5000 == 0:
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
                #     output_model_file = os.path.join(args.output_dir, 'model_step_%d.pt' % global_step)
                #     torch.save(model_to_save.state_dict(), output_model_file)
                #     output_config_file = os.path.join(args.output_dir, modeling.CONFIG_NAME)

            train_iter = train_iter_fct()
        time_to_train = time.time() - train_start
    if args.do_train and is_main_process() and not args.skip_checkpoint:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        output_model_file = os.path.join(args.output_dir, "bert_summarization.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, modeling.CONFIG_NAME)

        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_predict and (args.local_rank == -1 or is_main_process()):

        if not args.do_train and args.fp16:
            model.half()

        test_iter =data_loader.Dataloader(args, data_loader.load_dataset(args.predict_file, 'test', shuffle=False),
                                        args.predict_batch_size, device,
                                        shuffle=False, is_test=True)
        if args.device =="sdaa":
            with torch_sdaa.amp.autocast():
                calculate_rouge(args, model, test_iter)
        else:
            calculate_rouge(args, model, test_iter)


    if args.do_train:
        gpu_count = n_gpu
        if torch.distributed.is_initialized():
            gpu_count = torch.distributed.get_world_size()

        dllogger.log(step=tuple(), data={"e2e_train_time": time_to_train,
                                            "training_sequences_per_second": args.train_batch_size * args.gradient_accumulation_steps \
                                            * args.max_steps * gpu_count / time_to_train,
                                            "final_loss": final_loss})
    # if args.do_predict and is_main_process():
    #     dllogger.log(step=tuple(), data={"e2e_inference_time": time_to_infer,
    #                                              "inference_sequences_per_second": len(eval_features) / time_to_infer})
    # if args.do_eval and is_main_process():
    #     dllogger.log(step=tuple(), data={"exact_match": exact_match, "F1": f1})

if __name__ == "__main__":
    main()
    dllogger.flush()
