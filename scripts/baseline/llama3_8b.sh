torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset tvqa \
--blr 1e-2 --weight_decay 0.1 --accum_iter 4 --output_dir vqa_checkpoint/baseline/llama3_tvqa_1e2_acc4 --adapter_len 50 \
--llama3 --llama_model_path ./pretrained/llama3/ \

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 2e-2 --weight_decay 0.1 --accum_iter 4 --output_dir vqa_checkpoint/baseline/llama3_nextqa_2e2_acc4 --adapter_len 50 \
--llama3 --llama_model_path ./pretrained/llama3/ \

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 8B \
--max_seq_len 128 --batch_size 20 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset star \
--blr 2e-2 --weight_decay 0.1 --accum_iter 4 --output_dir vqa_checkpoint/baseline/llama3_star_2e2_acc4 --adapter_len 50 \
--llama3 --llama_model_path ./pretrained/llama3/ \

