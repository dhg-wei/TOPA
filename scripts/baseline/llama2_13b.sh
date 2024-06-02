torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 10 --warmup_epochs 4 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 1e-2 --weight_decay 0.1 --accum_iter 8 --output_dir vqa_checkpoint/baseline/llama2_13b_nextqa_1e2 --adapter_len 50 \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40 \

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset star \
--blr 2e-2 --weight_decay 0.1 --accum_iter 8 --output_dir vqa_checkpoint/baseline/llama2_13b_star_2e2 --adapter_len 50 \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40\


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 13B \
--max_seq_len 128 --batch_size 6 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset tvqa \
--blr 2e-2 --weight_decay 0.1 --accum_iter 8 --output_dir vqa_checkpoint/baseline/llama2_13b_vtqa_2e2 --adapter_len 50 \
--llama2 --llama_model_path ./pretrained/llama2/ --adapter_layer 40\