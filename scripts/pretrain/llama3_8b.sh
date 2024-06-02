randport=$(shuf -i8000-9999 -n1)  # Generate a random port number
torchrun --rdzv_endpoint 127.0.0.1:${randport} --nproc_per_node 4 train.py --model 8B \
--max_seq_len 150 --batch_size 14 --epochs 20 --warmup_epochs 1 --bias 3.5 --tau 100. --max_feats 10 --dataset textvid \
--blr 5e-3 --weight_decay 0.1 --output_dir ./vqa_checkpoint/checkpoint_pretrain/llama3_test --accum_iter 8 --textvid --variance 0.0 --memory --video_caption --vaq --openvqa --answer_balance --adapter_len 50 \
--llama3 --llama_model_path ./pretrained/llama3/ \
