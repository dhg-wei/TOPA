import torch
from .base_dataset import BaseDataset
import pandas as pd
import pickle
import numpy as np
import numpy
import random

class EgoSchema(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/egos/{split}.csv')

        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        
        self.features_path = './data/egos/features/'

        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?" and question[-1] != ".":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The correct choice is "
        open_options = [f"Answer: {option}" for option in options]
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options,'open_options':open_options}
        return text
        

    def _get_video(self, video):
        video=torch.from_numpy(video).float()
        video = video/video.norm(dim=-1,keepdim=True)

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
        while True:
            try:
                vid = self.data['uid'].values[idx]
        
                qtype = 1
                answer = self.data['answer'].values[idx]

                text = self._get_text(idx)


                text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
                q_index = len(self.tokenizer.sp_model.encode(text['q_text']))+1+video_start['vqa']+self.max_feats
                if self.args.openvqa_eval:
                    text['oa_text'] = f"_"
                    text_id_c, label_c, video_start_c, video_index_c, label_mask_c = self._get_openvqa_token(text, answer)
                    text_id.update(text_id_c)
                    label.update(label_c)
                    video_start.update(video_start_c)
                    video_index.update(video_index_c)
                    label_mask.update(label_mask_c)
                        
                v_path = f'{self.features_path}{vid}.npy'
                with open(v_path,'rb') as f:
                    video =  numpy.load(f)
                video, video_len = self._get_video(video)
                
                return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start, "video_index": video_index, "label_mask": label_mask, "qid": vid, "answer": answer, "qtype": qtype,"q_index": q_index}
            except:
                print(idx)
                idx = np.random.randint(0, len(self)-1)

    def __len__(self):
        return len(self.data)