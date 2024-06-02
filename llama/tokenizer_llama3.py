# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class Tokenizer_llama3:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.sp_model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.sp_model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

        self.v_token_id = 10955
        self.q_token_id = 14924
        self.a_token_id = 16533
        self.c_token_id = 5116
        self.nl_id = 627

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Choose the correct answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, video_start

    def encode_vaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.q_token_id) + 1
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.q_token_id) + 1
        return t, prefix_index, video_start

    
    def encode_qav(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the video based on the question and answer.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + q_text + o_text + a_text
        
        if split == 'train':
            s1 = s1 + answer_mapping[answer] + "\n" + "Video:"
            t1 = [self.bos_id] + self.sp_model.encode(s1)
            t = [t1 + [-2 for _ in range(max_feats)] + [self.eos_id]]
            prefix_index = t[0].index(self.v_token_id) + 1
        else:
            t = []
            for k, v in answer_mapping.items():
                t1 = [self.bos_id] + self.sp_model.encode(s1 + v + "\n" + "Video:") + [-2 for _ in range(max_feats)] + [self.eos_id]
                t.append(t1)
            prefix_index = t[answer].index(self.v_token_id) + 1
        return t, prefix_index

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def encode_dvqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        prefix_index = len(t[0]) - 4
        
        return t, prefix_index, video_start, prefix_i, prefix_main

    def encode_dvaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)
        
        prefix_index = t[0].index(self.q_token_id) + 1
        
        return t, prefix_index, video_start, prefix_i, prefix_main
    
    def encode_dqav(self, text=None, max_feats=10, max_seq_len=128, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the video based on the dialogue, question and answer.\n"
        d_text = text['d_text']
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        s1, s2, s3 = i_text, d_text, q_text + o_text + a_text

        t1 = [self.bos_id] + self.sp_model.encode(s1)
        t2 = self.sp_model.encode(s2)
        prefix_i, prefix_q = len(t1), len(t1) + len(t2)

        if split == 'train':
            t3 = self.sp_model.encode(s3 + answer_mapping[answer] + "\n" + "Video:")
            t = [t1 + t2 + t3 + [-2 for _ in range(max_feats)] + [self.eos_id]]
        else:
            t = []
            for k, v in answer_mapping.items():
                t3 = self.sp_model.encode(s3 + v + "\n" + "Video:") + [-2 for _ in range(max_feats)] + [self.eos_id]
                t.append(t1 + t2 + t3)
                
        prefix_index = len(t[0]) - max_feats - 1
        
        return t, prefix_index, prefix_i, prefix_q
    
    
    
    def encode_videocap(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Generate a dense description for the video.\n"
        # q_text = text['q_text'].strip()
        # o_text = text['o_text']
        # a_text = text['a_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        s2 = text['c_text']
        
        if split == 'train':
            # s2 = s2 + answer_mapping[answer] + "\n" + q_text
            s2 = "\n"+s2
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.c_token_id) + 1
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.c_token_id) + 1
        return t, prefix_index, video_start
    
    

    def encode_openvqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        oa_text = text['oa_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        if split == 'train':
            s2 = q_text + oa_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id)+1
        else:
            t = []
            for open_option in text['open_options']:
                s2 = q_text + open_option
                t2 = self.sp_model.encode(s2) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[0].index(self.a_token_id)+1
        return t, prefix_index, video_start