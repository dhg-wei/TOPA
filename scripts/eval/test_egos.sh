# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 8B \
# --max_seq_len 150 --batch_size 10 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset tvqa \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama3_7b_acc4_br5e3_ep20_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama3 --llama_model_path ./pretrained/llama3/ --memory \

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
--max_seq_len 128 --batch_size 15 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_correct_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
--output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
--llama2 --llama_model_path ./pretrained/llama2/ \
--evalall

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 15 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br1e2_ep20_vlastt/checkpoint_16.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --memory --test

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset tvqa \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_finetune/tag_vnips_7b_finetune_nextqa_5e3_acc8/checkpoint_1.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --evalall

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_finetune/tag_vnips_7b_finetune_realstar_1e2_acc4/checkpoint_4.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# -test
# --evalall


# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_finetune/tag_vnips_7b_finetune_tvqa_5e3_acc4/checkpoint_4.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --test
# --evalall

