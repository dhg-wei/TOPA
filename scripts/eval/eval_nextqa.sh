torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B --adapter_layer 40 \
--max_seq_len 128 --batch_size 10 --epochs 1 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.16 --resume vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs4_vnips/checkpoint_18.pth --adapter_len 50  --accum_iter 2 --eval \
--output_dir results/llama2_13B/nextqa \
--llama2 --llama_model_path ./pretrained/llama2/ \
--memory \


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 128 --batch_size 10 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.16 --resume vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_correct_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
--output_dir results/llama2_7B/nextqa \
--llama2 --llama_model_path ./pretrained/llama2/ \
--memory \


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 150 --batch_size 10 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.16 --resume vqa_checkpoint/checkpoint_pretrain/llama3_7b_acc4_br5e3_ep20_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
--output_dir results/llama3_8B/nextqa \
--llama3 --llama_model_path ./pretrained/llama3/ \
--memory \


