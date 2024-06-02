torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 5e-3 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs6/checkpoint_18.pth  --output_dir vqa_checkpoint/checkpoint_finetune/vnip_llama2_13b_finetune_nextqa_5e3_acc16 --accum_iter 16 --adapter_len 50 --finetune \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40 \


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset star \
--blr 5e-3 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs6/checkpoint_18.pth  --output_dir vqa_checkpoint/checkpoint_finetune/vnip_llama2_13b_finetune_star_5e3_acc16 --accum_iter 16 --adapter_len 50 --finetune \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40 \


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 5 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset tvqa \
--blr 5e-3 --weight_decay 0.1 --resume vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs6/checkpoint_18.pth  --output_dir vqa_checkpoint/checkpoint_finetune/vnip_llama2_13b_finetune_tvqa_5e3_acc16 --accum_iter 16 --adapter_len 50 --finetune \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40 \
