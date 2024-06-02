
# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 13B \
# --max_seq_len 128 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/wei/vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs4_ep20_vlast/checkpoint_15.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40 \
# --memory 

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 13B --adapter_layer 40 \
--max_seq_len 128 --batch_size 1 --epochs 1 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
--blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama2_13b_acc8_br8e3_bs4_vnips/checkpoint_18.pth --adapter_len 50  --accum_iter 2 --eval \
--output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
--llama2 --llama_model_path ./pretrained/llama2/ \
--test

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_correct_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --test


# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 13B --adapter_layer 40 \
# --max_seq_len 128 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 1 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --adapter_len 1  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --onlyqa --ori_llama --test 



# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/wei/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_ep20_vlastt/checkpoint_17.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --memory 

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 250 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/scratch/exp/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_correct_vnips/checkpoint_19.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --memory --openvqa_eval

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 250 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/wei/vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_ep20_wp2_correct_womcqa/checkpoint_13.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoin/debug \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --memory --test



# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 100 --batch_size 4 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset egos \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/wei/vqa_checkpoint/baseline/llama2_7b_nextqa_2e2_acc4_d415/checkpoint_9.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoint/eval/baseline_nextqa_egos \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# --test

# torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 train.py --model 7B \
# --max_seq_len 128 --batch_size 4 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset nextqa \
# --blr 9e-2 --weight_decay 0.16 --resume /home/users/nus/idmwyk/wei/vqa_checkpoint/baseline/llama2_7b_nextqa_2e2_acc4_d415/checkpoint_9.pth --adapter_len 50  --accum_iter 2 --eval \
# --output_dir /home/users/nus/idmwyk/wei/vqa_checkpoint/eval/baseline_nextqa \
# --llama2 --llama_model_path ./pretrained/llama2/ \
# # --test

