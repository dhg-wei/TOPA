
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 1e-2 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama3_7b_acc4_br5e3_ep20_vnips/checkpoint_19.pth  --output_dir vqa_checkpoint/checkpoint_finetune/llama3_finetune_nextqa_1e2_acc4 --accum_iter 4 --adapter_len 50 --finetune \
--llama3 --llama_model_path ./pretrained/llama3/  \


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset star \
--blr 1e-2 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama3_7b_acc4_br5e3_ep20_vnips/checkpoint_19.pth  --output_dir vqa_checkpoint/checkpoint_finetune/llama3_finetune_star_1e2_acc4_ep10 --accum_iter 4 --adapter_len 50 --finetune \
--llama3 --llama_model_path ./pretrained/llama3/  \



torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset tvqa \
--blr 5e-3 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama3_7b_acc4_br5e3_ep20_vnips/checkpoint_19.pth  --output_dir vqa_checkpoint/checkpoint_finetune/llama3_finetune_tvqa_5e3_acc4 --accum_iter 4 --adapter_len 50 --finetune \
--llama3 --llama_model_path ./pretrained/llama3/  \



