from typing import Dict, List, Optional
import torch
import warnings

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class T2TDataCollator():
    def __init__(self, tokenizer, model_type="t5", mode='training', using_tpu=False):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode
        self.using_tpu = using_tpu
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        self.decoder_start_token_id = self.pad_token_id or self.eos_token_id or 0

        # 安全检查
        if self.pad_token_id is None:
            warnings.warn("pad_token_id is None, using 0 as default.")
            self.pad_token_id = 0

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        # 调试信息
        # print(f"Batch size: {len(batch)}")
        # if len(batch) > 0:
        #     print(f"First example type: {type(batch[0])}")
        #     if isinstance(batch[0], (tuple, list)):
        #         print(f"First example length: {len(batch[0])}")
        #     elif hasattr(batch[0], 'keys'):
        #         print(f"First example keys: {list(batch[0].keys())}")
        
        try:
            # 处理元组格式 (source_ids, target_ids, attention_mask)
            if isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 3:
                input_ids = torch.stack([example[0] for example in batch])
                target_ids = torch.stack([example[1] for example in batch])
                attention_mask = torch.stack([example[2] for example in batch])
            else:
                raise ValueError(f"Unsupported batch format. First example: {batch[0]}")
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            raise e

        max_length = 512
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length] if attention_mask is not None else None
        target_ids = target_ids[:, :max_length]

        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        target_ids = torch.clamp(target_ids, -100, self.vocab_size - 1)  # labels 可含 -100

        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask)
            target_ids = trim_batch(target_ids, self.pad_token_id)

        # 构造 decoder_input_ids 和 lm_labels
        if self.model_type == "t5":
            lm_labels = target_ids.clone()
            decoder_input_ids = self._shift_right_t5(lm_labels)
            if self.mode == 'training':
                lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100)
        else:
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            if self.mode == 'training':
                lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100)

        self._validate_tensor("input_ids", input_ids, min_val=0)
        self._validate_tensor("decoder_input_ids", decoder_input_ids, min_val=0)
        self._validate_tensor("labels", lm_labels, min_val=-100, allow_neg100=True)

        # self._print_range("input_ids", input_ids)
        # self._print_range("labels", lm_labels)
        # self._print_range("decoder_input_ids", decoder_input_ids)

        # print(f"Generated batch shapes - input_ids: {input_ids.shape}, target_ids: {target_ids.shape}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }

    def _shift_right_t5(self, input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.decoder_start_token_id

        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, self.pad_token_id)

        shifted_input_ids = torch.clamp(shifted_input_ids, 0, self.vocab_size - 1)

        if not torch.all(shifted_input_ids >= 0):
            raise ValueError("decoder_input_ids contains negative values other than -100")

        return shifted_input_ids

    def _validate_tensor(self, name: str, tensor: torch.Tensor, min_val: int, allow_neg100: bool = False):
        if not isinstance(tensor, torch.Tensor):
            return

        if allow_neg100:
            valid_mask = (tensor == -100) | ((tensor >= 0) & (tensor < self.vocab_size))
        else:
            valid_mask = (tensor >= 0) & (tensor < self.vocab_size)

        if not torch.all(valid_mask):
            bad_vals = tensor[~valid_mask].unique()
            warnings.warn(f"{name} contains invalid token IDs: {bad_vals.tolist()}. Clamping...")
            if allow_neg100:
                tensor[~valid_mask] = -100
            else:
                tensor[~valid_mask] = self.pad_token_id  # 或 0

    def _print_range(self, name: str, tensor: torch.Tensor):
        """打印 tensor 的有效值范围"""
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return

        if tensor.numel() == 0:
            # print(f"{name} is empty")
            return

        if name == "labels":
            valid = tensor != -100
            if valid.any():
                t_valid = tensor[valid]
                # print(f"{name} range: {t_valid.min().item()} to {t_valid.max().item()}")
                pass
            else:
                # print(f"{name} all -100")
                pass
        else:
            # print(f"{name} range: {tensor.min().item()} to {tensor.max().item()}")
            pass