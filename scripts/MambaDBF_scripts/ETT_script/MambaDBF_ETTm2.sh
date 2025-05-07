export CUDA_VISIBLE_DEVICES=0

model_name=MambaDBF

dstate=16
lr=0.0001
e_layers=2
d_layers=1
seq_len=96
random_seed=2024

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --batch_size 32 \
  --label_len 96 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --train_epochs 30 \
  --patience 5 \
  --e_fact 4 \
  --d_model 128 \
  --d_ff 128 \
  --d_state 4 \
  --learning_rate $lr \
  --e_layers 1 \
  --d_layers 1 \
  --itr 1


python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --batch_size 32 \
  --label_len 96 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --train_epochs 30 \
  --e_fact 2 \
  --patience 5 \
  --d_model 128 \
  --d_ff 128 \
  --d_state 8 \
  --learning_rate $lr \
  --e_layers 1 \
  --d_layers 1 \
  --itr 1

  python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --batch_size 32 \
  --label_len 96 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --train_epochs 30 \
  --e_fact 2 \
  --patience 5 \
  --d_state 8 \
  --learning_rate $lr \
  --e_layers 1 \
  --d_layers 1 \
  --itr 1

  python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --batch_size 32 \
  --label_len 96 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --train_epochs 30 \
  --patience 5 \
  --d_state 8 \
  --e_fact 2 \
  --learning_rate $lr \
  --e_layers 1 \
  --d_layers 1 \
  --itr 1


