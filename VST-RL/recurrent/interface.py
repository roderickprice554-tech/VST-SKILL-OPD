# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type, List, Union, Dict, Tuple
from uuid import uuid4
import numpy as np

import torch
from tensordict import TensorDict
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.protocol import DataProto, DataProtoItem
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


RECURRENT_TURN_METADATA_KEYS = (
    "group_uid",
    "trajectory_uid",
    "sample_index",
    "transition_index",
    "previous_memory_tokens",
    "current_chunk_boundary",
    "generated_y_t_tokens",
    "updated_memory_tokens",
    "policy_version",
    "final_mask",
)

_TRANSITION_ONLY_KEYS = (
    "transition_index",
    "previous_memory_tokens",
    "generated_y_t_tokens",
    "updated_memory_tokens",
)


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def make_repeated_rollout_ids(group_uids, repeat_times: int):
    """Repeat GRPO group IDs while assigning a unique ID to each sampled rollout."""
    if not isinstance(repeat_times, int) or repeat_times < 1:
        raise ValueError("repeat_times must be a positive integer")

    group_uids = np.asarray(group_uids, dtype=object)
    if group_uids.ndim != 1 or len(set(group_uids.tolist())) != len(group_uids):
        raise ValueError("group_uids must be a one-dimensional array of unique IDs")

    repeated_group_uids = np.repeat(group_uids, repeat_times)
    trajectory_uids = np.asarray(
        [
            f"{group_uid}:rollout-{rollout_index}"
            for group_uid in group_uids
            for rollout_index in range(repeat_times)
        ],
        dtype=object,
    )
    return repeated_group_uids, trajectory_uids


def validate_recurrent_turns(
    output: DataProto,
    final_mask: torch.Tensor,
    sample_index: torch.Tensor,
) -> None:
    """Validate row alignment and the memory/final boundary of a rollout batch."""
    row_count = len(sample_index)
    if len(output) != row_count or len(final_mask) != row_count:
        raise ValueError("rollout tensors, final_mask, and sample_index must have equal row counts")

    for key in RECURRENT_TURN_METADATA_KEYS:
        if key not in output.non_tensor_batch:
            raise ValueError(f"missing recurrent turn metadata: {key}")
        if len(output.non_tensor_batch[key]) != row_count:
            raise ValueError(f"metadata {key} is not row-aligned")

    if "response_mask" not in output.batch or len(output.batch["response_mask"]) != row_count:
        raise ValueError("response_mask is not row-aligned")

    expected_sample_index = _as_list(sample_index)
    metadata_sample_index = _as_list(output.non_tensor_batch["sample_index"])
    if metadata_sample_index != expected_sample_index:
        raise ValueError("metadata sample_index does not match rollout sample_index")

    final_rows = _as_list(final_mask)
    metadata_final_rows = _as_list(output.non_tensor_batch["final_mask"])
    if metadata_final_rows != final_rows:
        raise ValueError("metadata final_mask does not match rollout final_mask")

    group_uids = _as_list(output.non_tensor_batch["group_uid"])
    uids = _as_list(output.non_tensor_batch["trajectory_uid"])
    transition_indices = _as_list(output.non_tensor_batch["transition_index"])
    policy_versions = _as_list(output.non_tensor_batch["policy_version"])
    if len(set(policy_versions)) != 1:
        raise ValueError("policy_version must be frozen for the complete rollout batch")

    rows_by_uid = {}
    for row, uid in enumerate(uids):
        rows_by_uid.setdefault(uid, []).append(row)

    for uid, rows in rows_by_uid.items():
        uid_group_uids = {group_uids[row] for row in rows}
        if len(uid_group_uids) != 1:
            raise ValueError(f"trajectory_uid {uid!r} maps to multiple group_uid values")
        if uid in uid_group_uids:
            raise ValueError("trajectory_uid must be distinct from group_uid")

        uid_sample_indices = {expected_sample_index[row] for row in rows}
        if len(uid_sample_indices) != 1:
            raise ValueError(f"trajectory_uid {uid!r} maps to multiple sample_index values")

        uid_final_rows = [row for row in rows if final_rows[row]]
        if len(uid_final_rows) != 1:
            raise ValueError(f"trajectory_uid {uid!r} must have exactly one final row")

        memory_rows = [row for row in rows if not final_rows[row]]
        actual_indices = sorted(transition_indices[row] for row in memory_rows)
        if actual_indices != list(range(len(memory_rows))):
            raise ValueError(f"trajectory_uid {uid!r} transition_index must be contiguous from zero")

        final_row = uid_final_rows[0]
        for key in _TRANSITION_ONLY_KEYS:
            if output.non_tensor_batch[key][final_row] is not None:
                raise ValueError(f"final row must set {key} to None")

    final_uids = [uids[row] for row, is_final in enumerate(final_rows) if is_final]
    if len(set(final_uids)) != len(final_uids):
        raise ValueError("each final row must have a unique trajectory_uid")


def aggregate_trajectories(
    output: DataProto,
    final_mask: torch.Tensor,
    sample_index: torch.Tensor,
) -> Dict[str, List[int]]:
    """Return ordered row indices for each complete recurrent trajectory."""
    validate_recurrent_turns(output, final_mask, sample_index)
    final_rows = _as_list(final_mask)
    uids = _as_list(output.non_tensor_batch["trajectory_uid"])
    transition_indices = _as_list(output.non_tensor_batch["transition_index"])
    trajectories = {}
    for uid in dict.fromkeys(uids):
        memory_rows = [row for row, row_uid in enumerate(uids) if row_uid == uid and not final_rows[row]]
        memory_rows.sort(key=lambda row: transition_indices[row])
        final_row = next(row for row, row_uid in enumerate(uids) if row_uid == uid and final_rows[row])
        trajectories[uid] = memory_rows + [final_row]
    return trajectories


def propagate_trajectory_reward(
    final_reward: torch.Tensor,
    output: DataProto,
    final_mask: torch.Tensor,
    sample_index: torch.Tensor,
) -> torch.Tensor:
    """Map final-only rewards to turns after validating UID/index identity."""
    validate_recurrent_turns(output, final_mask, sample_index)
    final_rows = _as_list(final_mask)
    row_sample_indices = _as_list(sample_index)
    uids = _as_list(output.non_tensor_batch["trajectory_uid"])

    final_uid_by_sample = {}
    for row, is_final in enumerate(final_rows):
        if is_final:
            final_uid_by_sample[row_sample_indices[row]] = uids[row]
    if sorted(final_uid_by_sample) != list(range(len(final_reward))):
        raise ValueError("final rows do not cover each final_reward sample_index exactly once")

    for row, sample in enumerate(row_sample_indices):
        if uids[row] != final_uid_by_sample[sample]:
            raise ValueError(
                f"trajectory_uid {uids[row]!r} does not match final row for sample_index {sample}"
            )

    return final_reward[sample_index.to(final_reward.device)]


@dataclass
class RConfig:
    """
    Configuration for Multi-turn Policy Optimization.
    Just an interface. Add anything you need in a subclass of it.
    """
    pass

class RDataset(RLHFDataset):
    """
    Dataset for Multi-turn Policy Optimization.
    This class can be used directly as a subclass of RLHFDataset for RecurrentRL
    (if you do not need any new features)

    Overwritten Method:
        - __getitem__: get a single sample
        - get_batch_keys: tensor keys and non-tensor keys, should be contained in the batch.
        - get_collate_fn: collate function for dataloader, default to the same as RLHFDataset.
    
    The inherited methods are hdfs/parquet related methods. 
    Make sure to call super().__init__() in your subclass to reuse RLHFDataset's initializer.
    """
    def __init__(
        self,
        recurrent_config: RConfig,
        data_files: Union[str, list[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=data_config, processor=processor)

    def __getitem__(self, item) -> dict:
        """
        Enforce subclass to override this method by declaring it as an abstract method.
        If you don't want to change its behavior, just return super().__getitem__(item).
        """
        row_dict = super().__getitem__(item)
        # used in validation metrics reduce
        row_dict["sample_uuid"] = str(uuid4())
        return row_dict


    def get_bactch_keys(self) -> tuple[list[str], list[str]]:
        return ["input_ids", "attention_mask", "position_ids"], []

    @staticmethod
    def get_collate_fn():
        return collate_fn

from .async_utils import ChatCompletionProxy

class AsyncOutput(ABC):
    def __init__(self, 
                 conversations: List[List[Dict[str, str]]], 
                 sample_index: int, 
                 final_mask: bool,
                 timing_raw: dict,
                 metrics: dict = None):
        self.conversations = conversations
        self.sample_index = sample_index
        self.final_mask = final_mask
        self.timing_raw = timing_raw
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        if "workflow/num_conv" not in metrics:
            metrics["workflow/num_conv"] = len(conversations)
    
class AsyncRAgent(ABC):
    """
    An async recurrent agent interface.

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM / How to process generated response / When to stop (rollout)
    > note that you should focus on a SINGLE sample instead of a group or a batch.
    """
    def __init__(self, proxy: ChatCompletionProxy, tokenizer:PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        self.proxy = proxy
        self.tokenizer = tokenizer
        self.config = config
        self.rollout_config = rollout_config
        self.timing_raw = {}

    # If you need to initialize/clean up some resource, override this two methods.
    def start(self, gen_batch: DataProto, timing_raw: dict):
        pass
    def end(self):
        pass
        

    @abstractmethod
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        Rollout a single sample, returns conversations/sample_index/final_mask + timing/metrics...
        """
        pass
    
    def sampling_params(self, meta_info):
        """
        Adapted from works/rollout/vllm_spmd_rollout, returns topp/temperature/n for generation
        Notice that you should specify max_completion_tokens manually, since it can be different for different agents
        Also notice that top_k is not supported in async mode
        """
        kwargs = dict(
                n=1,
                temperature=self.rollout_config.temperature,
                top_p=self.rollout_config.top_p,
            )
        do_sample = meta_info.get("do_sample", True)
        is_validate = meta_info.get("validate", False)
        if not do_sample:
                # logger.info(f"original {kwargs=}, updating becase do_sample is False")
            kwargs.update({
                    'best_of': 1,
                    'top_p': 1.0,
                    'min_p': 0.0,
                    'temperature': 0,
                    'n': 1  # if greedy, only 1 response
                })
        elif is_validate:
                # logger.info(f"original {kwargs=}, updating because is_validate is True")
                # TODO: try **
            kwargs.update({
                    'top_p': self.rollout_config.val_kwargs.top_p,
                    'temperature': self.rollout_config.val_kwargs.temperature,
                    'n': 1,  # if validate, already repeat in ray_trainer
                })
            
        return kwargs


    def reduce_timings(self, timing_raws: list[dict]) -> dict:
        """
        Reduce timing_raw of multiple agents.
        Make sure to follow the naming convention of timing_raw: "async" should be contained in the key,
        if and only if the timed code is an `await` statement.
        """
        reduced = {}
        for k in timing_raws[0]:
            if "async" in k:
                # async method can be executed parallelly
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws]) / len(timing_raws)
            else:
                # sync method is executed sequentially
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws])
        return reduced

    def reduce_metrics(self, metrics: list[dict]) -> dict:
        reduced = {}
        for k in metrics[0]:
            reduced[k + "_mean"] = np.mean([m[k] for m in metrics])
            reduced[k + "_min"] = np.min([m[k] for m in metrics])
            reduced[k + "_max"] = np.max([m[k] for m in metrics])
        return reduced

class RAgent(ABC):
    """
    A recurrent agent interface, you should focus on:

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM? (action)
    4. How to process generated response? (update)
    5. When to stop? (done)
    6. Any resource cleanup? (end)

    All methods are marked as abstract, they WILL NOT be called by default and are just a hint
    about how it should be implemented.
    """
    @abstractmethod
    def __init__(self, tokenizer:PreTrainedTokenizer, config: RConfig):
        pass
    @abstractmethod
    def start(self, gen_batch: DataProto, timing_raw: dict):
        """
        Called once at the beginning of generation loop.
        Initialize agent state, store gen_batch and timing_raw.
        """
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn to the sample id in the original batch
        pass
    @abstractmethod
    def action(self) -> tuple[list[torch.Tensor], dict]:
        """
        Called once for each rollout step.
        Return (input_ids(list[IntTensor]), meta_info).
        Remember to add sample_index to internal state.
        If the agent can decide if the sample is the final turn, also remember to add final_mask,
        else, you can decide in `update`.

        e.g. MemoryAgent will terminate the generation loop after all context is consumed, so it can
        compute a final_mask here
        """
        sample_index = torch.arange(len(self.gen_batch), dtype=torch.long)
        self.sample_index_list.append(sample_index)
        self.final_mask_list.append(torch.full(sample_index.shape, False, dtype=torch.bool))
        pass
    @abstractmethod
    def update(self, gen_output: DataProto) -> DataProto:
        """
        Called once after rollout, agent can execute tool calling or other custom action, and update agent state.
        
        e.g. CodeAgnet will terminate the generation loop if there is no code within ```python```.
        """
        pass
    @abstractmethod
    def done(self):
        """
        Whether the generation loop should stop.
        """
        return False
    @abstractmethod
    def end(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Called once after done() returns True.
        `del` the previouly saved data here, `gen_batch` for example.
        Can save some cpu memory(this batch will not be deleted until the next iteration).

        Returns final_mask(bool) and sample_index(long)
        """
        del self.gen_batch
        del self.timing_raw
        self.step = 0
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index

@dataclass
class RRegister:
    """Register your custom recurrent implementation with this class. The register object will be used to create these classes.
    """
    config_cls: Type[RConfig]
    dataset_cls: Type[RDataset]
    agent_cls: Type[RAgent]

    @classmethod
    def from_filename(cls, file_path: str, obj_name: str) -> 'RRegister':
        import importlib.util
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Recurrent implementation file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        if not spec:
            raise FileNotFoundError(f"Failed to create model spec for '{file_path}'.")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}")

        if not hasattr(module, obj_name):
            raise AttributeError(f"Register object '{obj_name}' not found in '{file_path}'.")

        obj = getattr(module, obj_name)
        if not isinstance(obj, cls):
            raise TypeError(f"Object '{obj_name}' in '{file_path}' is not an instance of {cls}.")
        print(f"[RECURRENT] recurrent enabled, using register '{obj_name}' from '{file_path}'.")
        return obj
