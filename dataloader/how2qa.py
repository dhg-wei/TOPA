import torch
from .base_dataset import BaseDataset
import pandas as pd
import pickle
import numpy as np
import numpy
class How2qa(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/how2qa/{split}.csv')

        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 4
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        
        self.features_path = '../../EgoSchema/benchmarking/FrozenBilm/features_how2qa/'

        print(f"Num {split} data: {len(self.data)}")
        
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
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video):
        video=torch.from_numpy(video).float()
        video = video/video.norm(dim=-1,keepdim=True)
        # video = torch.zeros(1, self.features_dim)
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

        return video, video_len

    def __getitem__(self, idx):
        while True:
            try:
            # if True:
                vid = self.data['uid'].values[idx]
                video_id =  self.data['video_id'].values[idx]
                qtype = 1
                answer = self.data['answer'].values[idx]
                text = self._get_text(idx)
                # print(text)
                # print(answer)
                text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)

                v_path = f'{self.features_path}{video_id}_{vid}.npy'
                # v_path = f'{self.features_path}507441ee-3eb4-4dc6-bac2-26bec2b66380.npy'
                with open(v_path,'rb') as f:
                    video =  numpy.load(f)
                video, video_len = self._get_video(video)

                # print(label_mask)
                return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                        "video_index": video_index, "label_mask": label_mask, "qid": vid, "answer": answer, "qtype": qtype}
            except:
                print(idx)
                idx = np.random.randint(0, len(self)-1)

    def __len__(self):
        return len(self.data)