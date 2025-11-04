# Adapted to tecorigin hardware

import os
import json
import torch
import time
import numpy as np
from tqdm import tqdm, trange
from typing import Dict, Optional, List, Union, Tuple, Any

from transformers import AutoTokenizer

from FlagEmbedding.evaluation.mldr.data_loader import MLDREvalDataLoader
from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs, AbsEvalModelArgs
from FlagEmbedding.abc.evaluation.evaluator import AbsEvaluator
from FlagEmbedding.abc.evaluation.searcher import EvalReranker

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def batch_pad(inputs, batch_size):
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    data_batch = input_ids.shape[0]
    if batch_size > data_batch:
        input_ids = torch.cat([input_ids] + [input_ids[-1:]] * (batch_size - data_batch))
        attention_mask = torch.cat([attention_mask] + [attention_mask[-1:]] * (batch_size - data_batch))

    return input_ids.numpy(), attention_mask.numpy(), data_batch

class BaseReranker():
    def __init__(
        self,
        pipeline,
        model_name_or_path: str,
        batch_size=256,
        max_length=512,
        query_max_length=512,
        normalize=False,
        rerank_top_k=100,
        cache_dir: Optional[str] = None,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir
        )
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.normalize = normalize
        self.rerank_top_k = rerank_top_k
        self.model_name = pipeline.model_name

    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        This is called during the reranking process.
        
        Parameters:
            corpus: Dict[str, Dict[str, Any]]: Corpus of documents. 
                Structure: {<docid>: {"text": <text>}}.
                Example: {"doc-0": {"text": "This is a document."}}
            queries: Dict[str, str]: Queries to search for.
                Structure: {<qid>: <query>}.
                Example: {"q-0": "This is a query."}
            search_results: Dict[str, Dict[str, float]]: Search results for each query.
                Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
                Example: {"q-0": {"doc-0": 0.9}}
            **kwargs: Any: Additional arguments.
        
        Returns: Dict[str, Dict[str, float]]: Reranked search results for each query. k is specified by rerank_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """
        # truncate search results to top_k
        for qid in search_results:
            search_results[qid] = dict(
                sorted(search_results[qid].items(), key=lambda x: x[1], reverse=True)[
                    :self.rerank_top_k
                ]
            )
        # generate sentence pairs
        sentence_pairs = []
        pairs = []
        for qid in search_results:
            for docid in search_results[qid]:
                if ignore_identical_ids and qid == docid:
                    continue
                sentence_pairs.append(
                    {
                        "qid": qid,
                        "docid": docid,
                        "query": queries[qid],
                        "doc": corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip(),
                    }
                )
                pairs.append(
                    (
                        queries[qid],
                        corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip()
                    )
                )
        # compute scores
        scores = self.compute_score(pairs)
        for i, score in enumerate(scores):
            sentence_pairs[i]["score"] = float(score)
        # rerank
        reranked_results = {qid: {} for qid in search_results}
        for pair in sentence_pairs:
            reranked_results[pair["qid"]][pair["docid"]] = pair["score"]
        return reranked_results
    
    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        **kwargs: Any
    ) -> List[float]:
        """_summary_

        Args:
            sentence_pairs (Union[List[Tuple[str, str]], Tuple[str, str]]): Input sentence pairs to compute scores.
            query_max_length (Optional[int], optional): Maximum length of tokens of queries. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            normalize (Optional[bool], optional): If True, use Sigmoid to normalize the results. Defaults to :data:`None`.

        Returns:
            List[float]: Computed scores of queries and passages.
        """
        if max_length is None: max_length = self.max_length
        if query_max_length is None: query_max_length = self.query_max_length
        if normalize is None: normalize = self.normalize

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        # tokenize without padding to get the correct length
        all_inputs = []
        batch_size = 3000
        for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize",
                                  disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                item = self.tokenizer.prepare_for_model(
                    q_inp,
                    d_inp,
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                )
                all_inputs.append(item)
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        np.save('all_inputs_sorted.npy', all_inputs_sorted)
        np.save('length_sorted_idx.npy', length_sorted_idx)
        # all_inputs_sorted = np.load('all_inputs_sorted.npy')
        # length_sorted_idx = np.load('length_sorted_idx.npy')

        e2e_start_time = time.time()
        time_info = {
                "data_load":[],
                "pre_time":[],
                "post_time":[],
                }

        if self.pipeline.target == "sdaa":
            self.pipeline.module.start_timing()
        else:
            time_info["infer_time"] = []

        all_scores = []
        start_data_time = time.time()
        for start_index in tqdm(range(0, len(all_inputs_sorted), self.batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < 128):
            sentences_batch = all_inputs_sorted[start_index:start_index + self.batch_size]
            time_info["data_load"].append(time.time() - start_data_time)
            start_time = time.time()

            inputs = self.tokenizer.pad(
                sentences_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            )

            input_ids, attention_mask, data_batch = batch_pad(inputs, self.batch_size)
            np.save(f'/mnt/usr/wanglm/25_q2/bge/debug_data/input_ids_{start_index}.npy', input_ids)
            np.save(f'/mnt/usr/wanglm/25_q2/bge/debug_data/attention_mask_{start_index}.npy', attention_mask)
            model_inputs = [input_ids, attention_mask]
            time_info["pre_time"].append(time.time() - start_time)
            
            scores = self.pipeline(model_inputs)[:data_batch]
            start_time = time.time()

            scores = torch.from_numpy(scores).float().view(-1, )
            all_scores.extend(scores.cpu().numpy().tolist())

            time_info["post_time"].append(time.time() - start_time)
            if self.pipeline.target != "sdaa":
                time_info["infer_time"].append(self.pipeline.run_time)
            start_data_time = time.time()

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]
        
        if self.pipeline.target == "sdaa":
            time_info["infer_time"] = [max(self.pipeline.module.get_infer_time()) / 1e3]
        time_info["e2e_time"] = time.time() - e2e_start_time

        batchs = len(all_inputs_sorted) // self.batch_size
        if len(all_inputs_sorted) % self.batch_size > 0:
            batchs += 1
        samples = batchs * self.batch_size

        print(f'summary: avg_sps: {samples / time_info["e2e_time"]}, e2e_time: {time_info["e2e_time"]}, data_time: {sum(time_info["data_load"])}, avg_inference_time: {sum(time_info["infer_time"]) / batchs}, avg_preprocess_time: {sum(time_info["pre_time"]) / batchs}, avg_postprocess: {sum(time_info["post_time"]) / batchs}')

        return all_scores

class MyAbsEvaluator(AbsEvaluator):
    def __call__(
        self,
        splits: Union[str, List[str]],
        search_results_save_dir: str,
        retriever="bge-m3",
        reranker: Optional[EvalReranker] = None,
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        dataset_name: Optional[str] = None,
        dataset_dir: str = None,
        **kwargs,
    ):
        """This is called during the evaluation process.

        Args:
            splits (Union[str, List[str]]): Splits of datasets.
            search_results_save_dir (str): Directory to save the search results.
            retriever (EvalRetriever): object of :class:EvalRetriever.
            reranker (Optional[EvalReranker], optional): Object of :class:EvalReranker. Defaults to :data:`None`.
            corpus_embd_save_dir (Optional[str], optional): Directory to save the embedded corpus. Defaults to :data:`None`.
            ignore_identical_ids (bool, optional): If True, will ignore identical ids in search results. Defaults to :data:`False`.
            k_values (List[int], optional): Cutoffs. Defaults to :data:`[1, 3, 5, 10, 100, 1000]`.
            dataset_name (Optional[str], optional): Name of the datasets. Defaults to :data:`None`.
        """
        # Check Splits
        checked_splits = self.data_loader.check_splits(splits, dataset_name=dataset_name)
        if len(checked_splits) == 0:
            print(f"{splits} not found in the dataset. Skipping evaluation.")
            return
        splits = checked_splits

        if dataset_name is not None:
            save_name = f"{dataset_name}-" + "{split}.json"
        else:
            save_name = "{split}.json"

        corpus_embd_save_dir = self.get_corpus_embd_save_dir(
            retriever_name=retriever,
            corpus_embd_save_dir=corpus_embd_save_dir,
            dataset_name=dataset_name
        )

        # Retrieval Stage
        no_reranker_search_results_save_dir = dataset_dir

        # from local
        no_reranker_search_results_dict = {}
        for split in splits:
            split_no_reranker_search_results_save_path = os.path.join(
                no_reranker_search_results_save_dir, '../', save_name.format(split=split)
            )
            data_info, search_results = self.load_search_results(split_no_reranker_search_results_save_path)

            self.check_data_info(
                data_info=data_info,
                model_name=retriever,
                reranker_name="NoReranker",
                split=split,
                dataset_name=dataset_name,
            )
            no_reranker_search_results_dict[split] = search_results

        # eval_results_save_path = os.path.join(no_reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
        # if not os.path.exists(eval_results_save_path) or self.overwrite or flag:
        #     retriever_eval_results = self.evaluate_results(no_reranker_search_results_save_dir, k_values=k_values)
        #     self.output_eval_results_to_json(retriever_eval_results, eval_results_save_path)

        # Reranking Stage
        if reranker is not None:
            reranker_search_results_save_dir = os.path.join(
                search_results_save_dir, retriever, reranker.model_name
            )
            os.makedirs(reranker_search_results_save_dir, exist_ok=True)

            corpus = self.data_loader.load_corpus(dataset_name=dataset_name)

            queries_dict = {
                split: self.data_loader.load_queries(dataset_name=dataset_name, split=split)
                for split in splits
            }

            flag = False
            for split in splits:
                rerank_search_results_save_path = os.path.join(
                    reranker_search_results_save_dir, save_name.format(split=split)
                )

                if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                    continue

                flag = True
                rerank_search_results = reranker(
                    corpus=corpus,
                    queries=queries_dict[split],
                    search_results=no_reranker_search_results_dict[split],
                    ignore_identical_ids=ignore_identical_ids,
                    **kwargs,
                )

                self.save_search_results(
                    eval_name=self.eval_name,
                    model_name=retriever,
                    reranker_name=reranker.model_name,
                    search_results=rerank_search_results,
                    output_path=rerank_search_results_save_path,
                    split=split,
                    dataset_name=dataset_name,
                )

            eval_results_save_path = os.path.join(reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
            if not os.path.exists(eval_results_save_path) or self.overwrite or flag:
                reranker_eval_results = self.evaluate_results(reranker_search_results_save_dir, k_values=k_values)
                self.output_eval_results_to_json(reranker_eval_results, eval_results_save_path)

class MLDREvalRunner():
    """
    Evaluation runner of MIRACL.
    """
    def __init__(
        self,
        reranker_pipeline,
        eval_args: AbsEvalArgs,
        model_args: AbsEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args

        self.retriever = 'bge-m3'
        self.reranker = self.load_retriever_and_reranker(reranker_pipeline)
        self.data_loader = self.load_data_loader()
        self.evaluator = self.load_evaluator()

    def load_data_loader(self) -> MLDREvalDataLoader:
        """Load the data loader instance by args.

        Returns:
            MLDREvalDataLoader: The MLDR data loader instance.
        """
        data_loader = MLDREvalDataLoader(
            eval_name=self.eval_args.eval_name, # 'mldr'
            dataset_dir=self.eval_args.dataset_dir, # './mldr/data'
            cache_dir=self.eval_args.cache_path,    # './cache/data'
            token=self.eval_args.token, # None
            force_redownload=self.eval_args.force_redownload,   # False
        )
        return data_loader
    
    def load_evaluator(self):
        """Load the evaluator for evaluation

        Returns:
            AbsEvaluator: the evaluator to run the evaluation.
        """
        evaluator = MyAbsEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator
    
    def run(self):
        """
        Run the whole evaluation.
        """
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

    
        for dataset_name in dataset_names:
            print(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.output_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values,
                dataset_name=dataset_name,
                dataset_dir=self.eval_args.dataset_dir
            )
        print(f"{self.eval_args.eval_name} evaluation on {dataset_names} completed.")

        print("Start computing metrics.")
        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics
        )
    
    def load_retriever_and_reranker(self, reranker_pipeline=None):

        reranker = BaseReranker(reranker_pipeline, 
                                self.model_args.reranker_name_or_path, 
                                self.model_args.reranker_batch_size, 
                                cache_dir=self.eval_args.cache_path,
                                max_length=self.model_args.reranker_max_length,
                                query_max_length=self.model_args.reranker_query_max_length,
                                normalize=self.model_args.normalize,
                                rerank_top_k=self.eval_args.rerank_top_k,
                                )

        return reranker

    
    @staticmethod
    def evaluate_metrics(
        search_results_save_dir: str,
        output_method: str = "markdown",
        output_path: str = "./eval_dev_results.md",
        metrics: Union[str, List[str]] = ["ndcg_at_10", "recall_at_10"]
    ):
        """Evaluate the provided metrics and write the results.

        Args:
            search_results_save_dir (str): Path to save the search results.
            output_method (str, optional): Output results to `json` or `markdown`. Defaults to :data:`"markdown"`.
            output_path (str, optional): Path to write the output. Defaults to :data:`"./eval_dev_results.md"`.
            metrics (Union[str, List[str]], optional): metrics to use. Defaults to :data:`["ndcg_at_10", "recall_at_10"]`.

        Raises:
            FileNotFoundError: Eval results not found
            ValueError: Invalid output method
        """
        eval_results_dict = {}
        for model_name in sorted(os.listdir(search_results_save_dir)):
            model_search_results_save_dir = os.path.join(search_results_save_dir, model_name)
            if not os.path.isdir(model_search_results_save_dir):
                continue
            for reranker_name in sorted(os.listdir(model_search_results_save_dir)):
                reranker_search_results_save_dir = os.path.join(model_search_results_save_dir, reranker_name)
                if not os.path.isdir(reranker_search_results_save_dir):
                    continue
                eval_results_path = os.path.join(reranker_search_results_save_dir, 'EVAL', "eval_results.json")
                if os.path.exists(eval_results_path):
                    eval_results = json.load(open(eval_results_path, encoding='utf-8'))
                else:
                    raise FileNotFoundError(f"Eval results not found: {eval_results_path}")

                if model_name not in eval_results_dict:
                    eval_results_dict[model_name] = {}
                eval_results_dict[model_name][reranker_name] = eval_results

        print("eval_metric:", eval_results_dict[model_name][reranker_name]['en-test']['ndcg_at_10'])


def eval_bge_reranker_v2_m3(reranker_pipeline, args):

    eval_args = AbsEvalArgs(eval_name='mldr', 
                        dataset_dir=args.data_path,    # './mldr/data', 
                        force_redownload=False, 
                        dataset_names=['en'], 
                        splits=['test'], 
                        corpus_embd_save_dir=os.path.join(args.data_path, '../corpus_embd'), # './mldr/corpus_embd', 
                        output_dir='./mldr/search_results', 
                        search_top_k=1000, 
                        rerank_top_k=100, 
                        cache_path=os.path.join(args.data_path, '../cache/data'), 
                        token=None, 
                        overwrite=True, 
                        ignore_identical_ids=False, 
                        k_values=[10, 100], 
                        eval_output_method='markdown', 
                        eval_output_path='./mldr/mldr_eval_results.md', 
                        eval_metrics=['ndcg_at_10'])
    
    model_args = AbsEvalModelArgs(embedder_name_or_path='BAAI/bge-m3', 
                            embedder_model_class=None, 
                            normalize_embeddings=True, 
                            pooling_method='cls', 
                            use_fp16=True, 
                            devices=['cpu'], 
                            query_instruction_for_retrieval=None, 
                            query_instruction_format_for_retrieval='{}{}', 
                            examples_for_task=None, 
                            examples_instruction_format='{}{}', 
                            trust_remote_code=False, 
                            reranker_name_or_path=args.config_path, 
                            reranker_model_class=None, 
                            reranker_peft_path=None, 
                            use_bf16=False, 
                            query_instruction_for_rerank=None, 
                            query_instruction_format_for_rerank='{}{}', 
                            passage_instruction_for_rerank=None, 
                            passage_instruction_format_for_rerank='{}{}', 
                            cache_dir='./cache/model', 
                            embedder_batch_size=3000, 
                            reranker_batch_size=args.batch_size, 
                            embedder_query_max_length=512, 
                            embedder_passage_max_length=512, 
                            reranker_query_max_length=512, 
                            reranker_max_length=args.shape,     # 512
                            normalize=False, 
                            prompt=None, 
                            cutoff_layers=None, 
                            compress_ratio=1, 
                            compress_layers=None)

    runner = MLDREvalRunner(
        reranker_pipeline,
        eval_args=eval_args,
        model_args=model_args,
    )

    runner.run()
