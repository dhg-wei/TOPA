import torch
from .base_dataset import BaseDataset
import pandas as pd
import json
import random
import pickle
import numpy as np
import math
import pickle
def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)

    x = x + (torch.randn(x.shape, device=x.device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    noise = (torch.randn(x.shape, device=x.device) * std)
    noise/=noise.norm(dim=-1,keepdim=True)
    x = x+noise*std
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=-1)

class TextVid(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        
        with open('./data/textvid/textvid.json','r') as f:
            self.data = json.load(f)
        self.feature_path = './data/textvid/features'
        # with open('./data/textvid/feature_small.pkl','rb') as f:
        #     self.feature = pickle.load(f)
        self.letter2number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, question,options):
        question = question.capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        # options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(len(options)):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The correct choice is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video):
        video=video.float()
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
    
    def _get_frame_feature(self, features,frames):
        start = 0
        video = []
        features = features/features.norm(dim=-1,keepdim=True)

        for frame in frames:
            feature = features[start:start+len(list(frame.keys()))]
            feature = feature.mean(dim=0,keepdim=True)
            feature = feature/feature.norm(dim=-1,keepdim=True)
            video.append(feature)
            start+=len(list(frame.keys()))
        video = torch.cat(video,dim=0)
        video = noise_injection(video, variance=self.args.variance)
        return video


    def __getitem__(self, idx):
        while True:
            try:

                video_meta = self.data[idx]
        
                vid = video_meta['idx']
                with open(f'{self.feature_path}/{vid}.pkl','rb') as f:
                    features = pickle.load(f)
                # features = self.feature[vid]
                video = self._get_frame_feature(features,video_meta['frames'])
    
                qtype = 1
                qa = random.choice(video_meta['QAs'])
                question = qa['question']

                # if question.strip().lower().split(' ')[0] =='what':
                #     if random.random()>0.5:
                #         qa = random.choice(video_meta['QAs'])
                #         question = qa['question']
                #         if question.strip().lower().split(' ')[0] =='what':
                #             raise
                
                answer = qa['answer']
                answer = self.letter2number[answer]
                options = qa['options']
                answer_text = options[answer]
                if self.args.answer_balance:
                    if not (("both" in answer_text.lower()) or ("all" in answer_text.lower()) or ('none' in answer_text.lower()) or ('and' in answer_text.lower())):
                        random.shuffle(options)
                        answer = options.index(answer_text)
    
                text = self._get_text(question,options)
        
        
                text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
                video, video_len = self._get_video(video)
        
                if self.args.video_caption:
                    caption = video_meta['global_video_caption']
                    text['c_text'] = f"Description: {caption}\n"
                    text_id_c, label_c, video_start_c, video_index_c, label_mask_c = self._get_caption_token(text, answer)
                    text_id.update(text_id_c)
                    label.update(label_c)
                    video_start.update(video_start_c)
                    video_index.update(video_index_c)
                    label_mask.update(label_mask_c)
        
                if self.args.openvqa and (random.random()>0.5):
                    if 'answer_open_ended' in qa.keys():
                        if ('both' not in answer_text.lower()) and ('all' not in answer_text.lower()):
                            answer_open_ended = qa['answer_open_ended'].strip()
                            text['oa_text'] = f"Answer: {answer_open_ended}\n"
                            text_id_c, label_c, video_start_c, video_index_c, label_mask_c = self._get_openvqa_token(text, answer)
                            text_id.update(text_id_c)
                            label.update(label_c)
                            video_start.update(video_start_c)
                            video_index.update(video_index_c)
                            label_mask.update(label_mask_c)
            
                return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                        "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}

            except:
                idx = np.random.randint(0, len(self)-1)
                # print(f'Error reading {idx}')

    def __len__(self):
        if self.args.debug:
            return len(self.data[:10000])
        else:
            return len(self.data)