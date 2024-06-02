import torch
from .base_dataset import BaseDataset
import pandas as pd
import pickle
import json
import numpy

class NextQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.split =split
        self.data = pd.read_csv(f'./data/nextqa/{split}.csv')


        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        

        self.features = torch.load(f'./data/{args.dataset}/clipvitl14.pth')
        print(f"Num {split} data: {len(self.data)}")
        if self.split=='train':
            self.train_ratio = int(len(self.data)*self.args.data_ratio)
        else:
            self.train_ratio = int(len(self.data)*1)
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The correct choice is "
        open_options = [f"\nAnswer: {option}" for option in options]

        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options,'open_options':open_options}
        return text

    
    def _get_video(self, video):
        video = video/video.norm(dim=-1,keepdim=True)
        # video = torch.zeros(1, self.features_dim)
            # video = video.repeat(,1)
        
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats
        if self.args.single_frame:
            video = video[::2]
            video=video.repeat_interleave(2,dim=0)
        return video, video_len

    def __getitem__(self, idx):
        idx = idx%self.train_ratio
        vid = self.data['video'].values[idx]
        qid = self.data['qid'].values[idx]
        # print(vid)
        qtype = self.qtype_mapping[self.data['type'].values[idx]]
        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)


        video = self.features[f'{vid}'].float()      
        video, video_len = self._get_video(video)

        if self.args.openvqa_eval:
            text['oa_text'] = f"_"
            text_id_c, label_c, video_start_c, video_index_c, label_mask_c = self._get_openvqa_token(text, answer)
            text_id.update(text_id_c)
            label.update(label_c)
            video_start.update(video_start_c)
            video_index.update(video_index_c)
            label_mask.update(label_mask_c)
                    
        # print(label_mask)
        return {"vid": str(vid), "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": qid, "answer": answer, "qtype": qtype}

    def __len__(self):
        if self.args.debug:
            return len(self.data[:2000])
        return len(self.data[:])
 