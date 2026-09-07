import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Optional
from uuid import uuid4
import copy
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template,chat_template_v2, now, unpad
from verl.protocol import DataProto
from verl.utils.dataset.vision_utils import process_video
from tqdm import tqdm
import uuid
import os
import pickle
import lmdb
import hashlib

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class VideoMemoryConfig(RConfig):
    
    max_prompt_length: int 
    video_clip_token_size: int  # Number of tokens representing one video clip (e.g. 256 visual tokens)
    max_memorization_length: int  # max number of tokens to memorize (text summary of video)
    max_video_clips: int  # max number of video clips to process
    max_final_response_length: int
    video_key: str # column name for the video tokens in the dataset
    prompt_type: str
    max_video_frame: int

    @property
    def max_raw_input_length(self):
        # Total input = Prompt + One Video Clip + Memory
        return self.max_prompt_length + self.video_clip_token_size + self.max_memorization_length

    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class VideoMemoryDataset(RDataset):
    def __init__(
        self,
        recurrent_config: RConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional = None,
        prog_video: Optional[bool] = False,
        use_cache: bool = False,
    ):
        if data_config.truncation != 'center':
            raise ValueError('VideoMemoryDataset only support center truncation')
        if processor is None:
            raise ValueError("VideoMemoryDataset requires a 'processor'.")

        data_config.max_prompt_length = recurrent_config.max_video_clips * recurrent_config.video_clip_token_size
        data_config.video_key = recurrent_config.video_key

        self.prog_video = prog_video
        self.use_cache = use_cache
        self.video_cache_dir = "/mnt/verl_lmdb_cache" 
        self.env = None

        if self.prog_video and self.use_cache:
            logger.warning("⚠️ [VideoMemoryDataset] prog_video enabled. Cache DISABLED to ensure consistency.")
            self.use_cache = False

        self.env = None
        if self.use_cache:
            os.makedirs(self.video_cache_dir, exist_ok=True)

        if self.prog_video:
            self.prog_start_step = 0
            self.prog_end_step = 200
            self.min_frames = 80
            self.max_frames = recurrent_config.max_video_frame
            self.frame_step_size = 20
            
            self._sync_file = f"/tmp/verl_curriculum_{uuid4().hex}.txt"
            self._write_frames(self.min_frames)
            logger.info(f"Video Curriculum Enabled. Sync file: {self._sync_file}")
        else:
            self._sync_file = None
            self.current_max_frames = recurrent_config.max_video_frame 

        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    def _write_frames(self, frames: int):
        try:
            with open(self._sync_file, 'w') as f:
                f.write(str(frames))
        except Exception as e:
            logger.warning(f"Failed to write curriculum file: {e}")

    def _read_frames(self) -> int:
        if not self._sync_file or not os.path.exists(self._sync_file):
            return self.min_frames
        try:
            with open(self._sync_file, 'r') as f:
                val = f.read().strip()
                return int(val) if val else self.min_frames
        except Exception:
            return self.min_frames

    def _init_env(self):
        if self.env is None and self.use_cache:
            try:
                self.env = lmdb.open(
                    self.video_cache_dir, 
                    map_size=1099511627776, 
                    readonly=False, 
                    lock=True,       
                    readahead=False, 
                    meminit=False,
                    max_dbs=1
                )
            except Exception as e:
                logger.error(f"LMDB Init Failed: {e}")
                self.use_cache = False

    def set_step(self, global_step: int):
        if not self.prog_video:
            return
        
        if global_step < self.prog_start_step:
            target = self.min_frames
        elif global_step >= self.prog_end_step:
            target = self.max_frames
        else:
            progress = (global_step - self.prog_start_step) / (self.prog_end_step - self.prog_start_step)
            raw = self.min_frames + progress * (self.max_frames - self.min_frames)
            steps = round((raw - self.min_frames) / self.frame_step_size)
            target = self.min_frames + steps * self.frame_step_size
            target = min(max(target, self.min_frames), self.max_frames)
        
        target = int(target)
        current_in_file = self._read_frames()
        
        if target != current_in_file:
            logger.info(f"[Video Curriculum] Step {global_step}: Updating frames {current_in_file} -> {target}")
            self._write_frames(target)

    def _get_cache_key(self, index: int, video_path_info: str) -> bytes:
        path_hash = hashlib.md5(video_path_info.encode('utf-8')).hexdigest()
        return f"{index}_{path_hash}".encode('ascii')

    def __getitem__(self, item):
        
        self._init_env()
        raw_row = self.dataframe[item]
        video_paths_raw = raw_row.get(self.video_key, "")
        if isinstance(video_paths_raw, list):
            video_paths_str = "".join(video_paths_raw)
        else:
            video_paths_str = str(video_paths_raw)

        if self.prog_video:
            dynamic_frames = self._read_frames()
        else:
            dynamic_frames = self.current_max_frames

        if self.use_cache and self.env:
            key = self._get_cache_key(item, video_paths_str)
            try:
                # Read-only LMDB transaction for fast cache lookup.
                with self.env.begin(write=False) as txn:
                    data_bytes = txn.get(key)
                    if data_bytes:
                        cached_data = pickle.loads(data_bytes)
                        cached_data["sample_uuid"] = str(uuid4())
                        return cached_data
            except Exception as e:
                logger.warning(f"LMDB Read Error item {item}: {e}")

        row_dict: dict = copy.deepcopy(self.dataframe[item]) 

        chat = row_dict.pop(self.prompt_key)
        question = row_dict.pop('question')
        
        multi_modal_data = {}

        if self.video_key in row_dict:
            video_paths = row_dict.pop(self.video_key)
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            
            processed_video_list = []
            for video_path in video_paths:
                video_tensor = process_video(
                    {"video": video_path}, 
                    fps=2, 
                    fps_max_frames=dynamic_frames, 
                    max_pixels=160 * 28 * 28
                )
                if isinstance(video_tensor, torch.Tensor):
                    video_array = video_tensor.numpy().astype(np.uint8)
                else:
                    video_array = video_tensor.astype(np.uint8)
                processed_video_list.append(video_array)

            multi_modal_data["video"] = processed_video_list
            
        t, _, h, w = multi_modal_data["video"][0].shape
        context_len = int((t + 1) // 2 * h // 28 * w // 28)
        
        input_ids = torch.tensor([[151656] * context_len])
        attention_mask = torch.tensor([[1] * context_len])

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False, 
            truncation=self.truncation,
        )

        row_dict["context_ids"] = input_ids
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(chat, add_special_tokens=False)
        row_dict["question_ids"] = self.tokenizer.encode(question, add_special_tokens=False)
        
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = {
            'video_grid_thw': torch.tensor([[t, int(h//28), int(w//28)]]),
            'second_per_grid_ts': [1.0]
        }

        extra_info = row_dict.get("extra_info", {})
        row_dict["video_duration"] = extra_info.get('duration', 0.0)
        row_dict["index"] = extra_info.get("index", 0)
        row_dict["sample_uuid"] = str(uuid4())

       # Stage 3: write to LMDB cache.
        if self.use_cache and self.env:
            try:
                with self.env.begin(write=True) as txn:
                    txn.put(key, pickle.dumps(row_dict))
                
                # Resolve current worker id for logging.
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    w_id = "Main"
                else:
                    w_id = f"Worker-{worker_info.id}"
                
                # Include worker id in cache build logs.
                logger.info(f"[{w_id}] Build Disk Cache: {item} {self.video_cache_dir}")
                
            except lmdb.MapFullError:
                logger.warning("⚠️ LMDB is full! Cache disabled for this item.")
            except Exception as e:
                logger.warning(f"LMDB Write Error item {item}: {e}")

        return row_dict

    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
        return ["context_ids", "context_length"], ["prompt_ids", "question_ids"]

# Modified Template for Video Context
TEMPLATE_TYPE_1 = """{TimeStamp} {VideoClip}"""

TEMPLATE_TYPE_2 = """{TimeStamp} {VideoClip}
**Streaming Thinking Rules:**
1. **Update Only**: Observe the video segment, read previous text, and only record **new** facts from the current segment. Do not repeat history.
2. **Wait for End**: Do not provide the final answer until the video stream is complete. Currently, just accumulate evidence.

Start Analysis:
"""
TEMPLATE_FINAL_BOXED_TYPE_1 = """{TimeStamp} {VideoClip}
{EndTime} Based on the provided Video Memory and the Current Video Clip, answer the following Problem.
{PromptFinal}
Output the final answer in \\boxed{{}}.
Your answer:
"""
TEMPLATE_FINAL_BOXED_TYPE_2 = """{TimeStamp} {VideoClip}
{EndTime} Based on the provided Video Memory and the Current Video Clip, answer the following Problem. You must combine the context from the memory with the visual details of the current clip. Output the final answer in \\boxed{{}}.

{PromptFinal}

Your answer:
"""

MEMORY_PROMPT = """[System]
You are a Streaming Video Analyst.
{memory}"""


class VideoMemoryAgent(RAgent):
    def __init__(self, 
                tokenizer:PreTrainedTokenizer,
                config: VideoMemoryConfig,
                processor: Optional[ProcessorMixin] = None
                ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_template = chat_template_v2(processor)
        
        # Initialize templates

        if self.config.prompt_type == "type1":
            self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_TYPE_1,previous=MEMORY_PROMPT), tokenizer)
            self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED_TYPE_1,previous=MEMORY_PROMPT), tokenizer)
        elif self.config.prompt_type == "type2":
            self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_TYPE_2,previous=MEMORY_PROMPT), tokenizer)
            self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED_TYPE_2,previous=MEMORY_PROMPT), tokenizer)
        
        self.max_input_length = self.config.max_raw_input_length + max(self.token_message_template.length, self.token_final_message_template.length)
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {max(self.token_message_template.length, self.token_final_message_template.length)}(message_template) = {self.max_input_length}\n')
        
        self.NO_MEMORY_TOKENS = tokenizer.encode("", add_special_tokens=False)
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict): # Initialize agent state.
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] 
        self.sample_index_list = [] 
        
        # context_length here refers to the number of valid video tokens
        self.ctx_length = gen_batch.batch['context_length']
        self.bsz = len(self.ctx_length)
        tokens_per_frame = []
        num_frames = []
        for i in range(self.bsz):
            t, _, h, w = gen_batch.non_tensor_batch['multi_modal_data'][i]['video'][0].shape
            tokens = int((h/28) * (w/28))
            tokens_per_frame.append(tokens)
            num_frames.append(t)
        self.tokens_per_frame = torch.tensor(
            tokens_per_frame, 
            device=self.ctx_length.device,
            dtype=torch.long)
        self.num_frames = torch.tensor(
            num_frames,
            device=self.ctx_length.device,
            dtype=torch.long)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # 1) Determine current state.
        # active_mask checks if we still have video clips to process
        active_mask = self.ctx_length > (self.step + 1) * self.config.video_clip_token_size
        self.active_mask = active_mask
        
        # Decide whether to enter final turn.
        is_final_turn = (active_mask.sum().item() == 0)
        self.is_final = is_final_turn

        # 2) Select parameters based on current turn.
        if is_final_turn:
            # Final mode: process all samples with the final template.
            calc_step = self.step
            target_indices = list(range(self.bsz))
            template = self.token_final_message_template
        else:
            # Normal mode: process only active samples with the standard template.
            calc_step = self.step
            target_indices = torch.nonzero(active_mask).squeeze(1).tolist()
            template = self.token_message_template

        # 3) Prepare vectorized batch fields.
        batch_data = self.gen_batch.non_tensor_batch
        durations = batch_data['video_duration']
        mm_data = batch_data['multi_modal_data']

        if is_final_turn:
            prompts = batch_data['prompt_ids']

        if not is_final_turn:
            target_start = self.config.video_clip_token_size * calc_step
            target_end = self.config.video_clip_token_size * (calc_step + 1)
            
            start_frame_idx = (torch.floor(target_start / self.tokens_per_frame) * 2).int()
            end_frame_idx = (torch.floor(target_end / self.tokens_per_frame) * 2).int()
        else:
            target_start = (self.ctx_length // self.config.video_clip_token_size) * self.config.video_clip_token_size
            target_end = self.ctx_length
            start_frame_idx = (torch.floor(target_start / self.tokens_per_frame) * 2).int()
            end_frame_idx = (torch.floor(target_end / self.tokens_per_frame) * 2).int()

            empty_mask = (start_frame_idx == end_frame_idx)
            
            if empty_mask.any():
                MIN_OFFSET = 4 
                adjusted_start = end_frame_idx[empty_mask] - MIN_OFFSET   
                start_frame_idx[empty_mask] = torch.maximum(torch.tensor(0, device=adjusted_start.device), adjusted_start)

                
        # 5) Main single-pass loop.
        self.video_messages = []
        self.messages = []
        self.video_inputs = []
        self.batch_uids = []
        self.pending_turn_metadata = {
            'trajectory_uid': [],
            'sample_index': [],
            'transition_index': [],
            'previous_memory_tokens': [],
            'current_chunk_boundary': [],
        }

        for idx in tqdm(target_indices):
            # A) Slice current video segment.
            s_idx, e_idx = start_frame_idx[idx].item(), end_frame_idx[idx].item()

            raw_msg = batch_data['multi_modal_inputs'][idx]
            self.batch_uids.append(batch_data['uid'][idx])
            s_id = self.tokens_per_frame[idx] * 2 * s_idx 
            e_id = self.tokens_per_frame[idx] * 2 * e_idx
            vgw = raw_msg['video_grid_thw'].clone()
            vgw[0,0] = int((e_idx - s_idx)/2)
            vid_message = {
                'video_grid_thw':vgw,
                'second_per_grid_ts':raw_msg['second_per_grid_ts'],
            }
            self.video_inputs.append(vid_message)

            video_clip = mm_data[idx].copy()
            video_clip['video'] = list(mm_data[idx]['video']) 
            video_clip['video'][0] = video_clip['video'][0][s_idx:e_idx]
            self.video_messages.append(video_clip)

            # B) Compute clip timestamps.
            t_factor = durations[idx] / self.num_frames[idx]
            s_time = s_idx * t_factor
            e_time = e_idx * t_factor

            self.pending_turn_metadata['trajectory_uid'].append(batch_data['uid'][idx])
            self.pending_turn_metadata['sample_index'].append(idx)
            self.pending_turn_metadata['transition_index'].append(None if is_final_turn else self.step)
            previous_memory = None
            if not is_final_turn:
                previous_memory = list(self.memory[idx]) if self.memory[idx] is not None else []
            self.pending_turn_metadata['previous_memory_tokens'].append(previous_memory)
            self.pending_turn_metadata['current_chunk_boundary'].append({
                'frames': [s_idx, e_idx],
                'seconds': [float(s_time), float(e_time)],
            })
            
            ts_str = f"Time={s_time:.1f}-{e_time:.1f}s"
            ts_tokens = self.tokenizer.encode(ts_str, add_special_tokens=False)

            # C) Build template kwargs.
            vid_pad_num = (e_idx - s_idx + 1) // 2 * self.tokens_per_frame[idx]
            fmt_kwargs = {
                'memory': self.memory[idx] if self.memory[idx] is not None else self.NO_MEMORY_TOKENS,
                'TimeStamp': ts_tokens,
                'VideoClip': torch.tensor([151652] + [151656] * vid_pad_num + [151653]),
            }

            # Add final-only fields when in final mode.
            if is_final_turn:
                end_ts_str = f"Time={e_time:.1f}s"
                fmt_kwargs['EndTime'] = self.tokenizer.encode(end_ts_str, add_special_tokens=False)
                fmt_kwargs['PromptFinal'] = prompts[idx]

            # D) Render tokenized message.
            self.messages.append(template.format(**fmt_kwargs))

        # 6) Finalize output containers.
        self.video_messages = np.array(self.video_messages)
        self.batch_uids = np.array(self.batch_uids)
        
        # Build sample_index from processed indices.
        sample_index = torch.tensor(target_indices, dtype=torch.long, device=active_mask.device)
        final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) if is_final_turn else torch.full(sample_index.shape, False, dtype=torch.bool) # all False
        
        self.meta_info = {
            'max_video_token_num': self.config.video_clip_token_size + 500,
            'input_pad_to': self.max_input_length,
            'pad_to': self.config.gen_pad_to,
            'generation_kwargs': {
                'max_tokens': self.config.gen_pad_to,
                'n': 1
            }
        }
        
        log_msg = 'FINAL TURN: VideoMemoryAgent.next() done' if is_final_turn else f'VideoMemoryAgent.action() done (Step {self.step})'
        logger.info(log_msg)

        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        
        return self.messages, self.video_messages, self.meta_info, self.video_inputs, self.batch_uids


    def _get_time_stamp_ids(self, gen_output):
        TIME_ID = 1462
        END_ID = 151652
        
        # Adjust these pad token ids if tokenizer settings differ.
        # For Qwen-like tokenizers, common values are 151643/151655.
        PAD_TOKEN_1 = 151643  
        PAD_TOKEN_2 = 151656  

        out = []
        
        for item in gen_output.batch['prompts']:
            # 0) Normalize input sequence to a Python list.
            if hasattr(item, 'tolist'):
                raw_seq = item.tolist()
            else:
                raw_seq = item
            
            # 1) Remove both pad tokens while preserving order.
            clean_seq = [x for x in raw_seq if x != PAD_TOKEN_1 and x != PAD_TOKEN_2]
            
            start_pos = -1
            end_pos = -1
            seq_len = len(clean_seq)
            for i in range(seq_len - 1, -1, -1):
                if clean_seq[i] == END_ID:
                    end_pos = i
                    break
            
            # If END_ID is found, search backward for the nearest TIME_ID.
            if end_pos != -1:
                for i in range(end_pos - 1, -1, -1):
                    if clean_seq[i] == TIME_ID:
                        start_pos = i
                        break  # Stop at the nearest TIME_ID for the shortest match.
            
            if start_pos != -1 and end_pos != -1:
                segment = clean_seq[start_pos : end_pos]
                out.append(segment)
            else:
                # No valid pair found.
                out.append([])

        return np.array(out, dtype=object)

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        generated_tokens = [None] * len(self.pending_turn_metadata['sample_index'])
        updated_memory_tokens = [None] * len(self.pending_turn_metadata['sample_index'])
        if not self.is_final:
            # Update memory with the text description/summary of the video clip
            time_stamp = self._get_time_stamp_ids(gen_output)
            raw_responses = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            re_map_id = (self.active_mask.int().cumsum(dim=0)-1).tolist()
            for i, is_active in enumerate(self.active_mask):
                if is_active:
                    ts_item = time_stamp[re_map_id[i]]
                    if hasattr(ts_item, 'tolist'): 
                        ts_item = ts_item.tolist()
                    elif isinstance(ts_item, np.ndarray):
                        ts_item = ts_item.tolist()
                        
                    resp_item = raw_responses[re_map_id[i]]
                    if hasattr(resp_item, 'tolist'): 
                        resp_item = resp_item.tolist()
                    elif isinstance(resp_item, np.ndarray):
                        resp_item = resp_item.tolist()
                    
                    new_content = ts_item + resp_item + [198]
                    
                    if self.memory[i] is None:
                        self.memory[i] = new_content
                    else:
                        self.memory[i] += new_content

                    output_idx = re_map_id[i]
                    generated_tokens[output_idx] = list(resp_item)
                    updated_memory_tokens[output_idx] = list(self.memory[i])

        def object_array(items):
            result = np.empty(len(items), dtype=object)
            result[:] = items
            return result

        gen_output.non_tensor_batch.update({
            'trajectory_uid': object_array(self.pending_turn_metadata['trajectory_uid']),
            'sample_index': np.asarray(self.pending_turn_metadata['sample_index'], dtype=np.int64),
            'transition_index': object_array(self.pending_turn_metadata['transition_index']),
            'previous_memory_tokens': object_array(self.pending_turn_metadata['previous_memory_tokens']),
            'current_chunk_boundary': object_array(self.pending_turn_metadata['current_chunk_boundary']),
            'generated_y_t_tokens': object_array(generated_tokens),
            'updated_memory_tokens': object_array(updated_memory_tokens),
        })

        self.log_step(gen_output)
        self.step += 1
        return gen_output
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        def clip_long_string(string, max_length=2000):
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]
        
        def rm_video_pad(input_id):
            video_pad_token = 151656
            result = []
            for token in input_id:
                # Collapse consecutive video pad tokens.
                if token == video_pad_token and result and result[-1] == video_pad_token:
                    continue
                result.append(token)
            
            return result
            
        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT VIDEO] STEP{step}{'='*30}")

        # if self.active_mask[0]:
        decoded_message = self.tokenizer.decode(rm_video_pad(self.messages[0]))
        rsp0 = gen_output.batch['responses'][0]
        decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
        logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
        logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
        logger.info(f"[RESPONSE] {decoded_response}")
        logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")

REGISTER = RRegister(config_cls=VideoMemoryConfig, dataset_cls=VideoMemoryDataset, agent_cls=VideoMemoryAgent)
