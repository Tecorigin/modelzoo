# Adapted to tecorigin hardware

from typing import cast, List, Dict, Union

import numpy as np
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from mteb import MTEB

class FlagDRESModel:
    def __init__(
            self,
            pipeline,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            opt = None
    ) -> None:

        self.pipeline = pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = opt.batch_size
        self.opt = opt
        self.batchs = 0

        self.time_info = {
                "data_load":[],
                "pre_time":[],
                "post_time":[],
                }

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)


    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)


    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:

        all_embeddings = []
        start_data_time = time.time()
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            self.time_info["data_load"].append(time.time() - start_data_time)
            start_time = time.time()

            inputs = self.tokenizer(
                sentences_batch,
                padding="max_length",
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            input_ids, attention_mask, data_batch = self.batch_pad(inputs)
            model_inputs = [input_ids, attention_mask]
            self.time_info["pre_time"].append(time.time() - start_time)

            token_embeddings, _ = self.pipeline(model_inputs)
            start_time = time.time()

            last_hidden_state = torch.from_numpy(token_embeddings[:data_batch]).float()
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

            self.time_info["post_time"].append(time.time() - start_time)
            if self.opt.target != "sdaa":
                self.time_info["infer_time"].append(self.pipeline.run_time)
            self.batchs += 1
            start_data_time = time.time()

        return np.concatenate(all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

    def batch_pad(self, inputs):
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        data_batch = input_ids.shape[0]
        if self.batch_size > data_batch:
            input_ids = torch.cat([input_ids] + [input_ids[-1:]] * (self.batch_size - data_batch))
            attention_mask = torch.cat([attention_mask] + [attention_mask[-1:]] * (self.batch_size - data_batch))

        return input_ids.numpy(), attention_mask.numpy(), data_batch

def eval_run(model, evaluation, opt):

    if opt.target == "sdaa":
        model.pipeline.module.start_timing()
    else:
        model.time_info["infer_time"] = []
    e2e_start_time = time.time()

    result = evaluation.run(model, overwrite_results=True, output_folder=f"results/{opt.model_name}")
    
    if opt.target == "sdaa":
        model.time_info["infer_time"] = [max(model.pipeline.module.get_infer_time()) / 1e3]
    model.time_info["e2e_time"] = time.time() - e2e_start_time

    samples = model.batchs * opt.batch_size
    print(f'summary: avg_sps: {samples / model.time_info["e2e_time"]}, e2e_time: {model.time_info["e2e_time"]}, data_time: {sum(model.time_info["data_load"])}, avg_inference_time: {sum(model.time_info["infer_time"]) / model.batchs}, avg_preprocess_time: {sum(model.time_info["pre_time"]) / model.batchs}, avg_postprocess: {sum(model.time_info["post_time"]) / model.batchs}')

    return result

def eval_bge_large_zh_v15(pipeline, opt):

    model = FlagDRESModel(pipeline,
                          model_name_or_path=opt.config_path,
                          pooling_method='cls',
                          opt=opt)

    evaluation = MTEB(tasks=['Ocnli'])
    evaluation.tasks[0].metadata.dataset['path'] = opt.data_path
    
    result = eval_run(model, evaluation, opt)
    print("eval_metric:", result[0].scores['validation'][0]['cosine_ap'])

def eval_bge_large_en_v15(pipeline, opt):

    model = FlagDRESModel(pipeline,
                          model_name_or_path=opt.config_path,
                          normalize_embeddings=False,
                          pooling_method='cls',
                          opt=opt)

    task = 'NFCorpus'
    evaluation = MTEB(tasks=[task], task_langs=['en'], eval_splits = ["test" if task not in ['MSMARCO'] else 'dev'])
    evaluation.tasks[0].metadata.dataset['path'] = opt.data_path

    result = eval_run(model, evaluation, opt)
    print("eval_metric:", result[0].to_dict()['scores']['test'][0]['main_score'])

def eval_bge_m3(pipeline, opt):
    eval_bge_large_en_v15(pipeline, opt)