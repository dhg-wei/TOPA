import torch
from .base_dataset import BaseDataset
import json

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        self.features = torch.load(f'./data/star/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        self.split = split
        print(f"Num {split} data: {len(self.data)}") 
        if split=='train':
            self.train_ratio = int(len(self.data)*self.args.data_ratio)
        else:
            self.train_ratio = int(len(self.data))

    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        answer = options.index(self.data[idx]['answer'])
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        open_options = [f"Answer: {option}" for option in options]
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options,'open_options':open_options}
        return text, answer

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start: end +1, :].float() # ts
        video = video/video.norm(dim=-1,keepdim=True)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        idx = idx%self.train_ratio
        vid = self.data[idx]['video_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        video, video_len = self._get_video(f'{vid}', start, end)

        if self.args.openvqa_eval:
            text['oa_text'] = f"_"
            text_id_c, label_c, video_start_c, video_index_c, label_mask_c = self._get_openvqa_token(text, answer)
            text_id.update(text_id_c)
            label.update(label_c)
            video_start.update(video_start_c)
            video_index.update(video_index_c)
            label_mask.update(label_mask_c)
            
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}


    def __len__(self):
        return len(self.data)