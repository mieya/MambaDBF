export CUDA_VISIBLE_DEVICES=1

model_name=MambaDBF
dstate=16
seq_len=192
e_fact=2
random_seed=2024

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 2 \
  --label_len 96 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --d_state 16 \
  --d_ff 128 \
  --d_model 128 \
  --train_epochs 50 \
  --e_layers 2 \
  --d_layers 2 \
  --patience 5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1



python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 1 \
  --label_len 96 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --d_state 16 \
  --d_ff 128 \
  --d_model 128 \
  --des 'Exp' \
  --train_epochs 30 \
  --patience 7 \
  --e_layers 2 \
  --d_layers 2 \
  --learning_rate 0.00005 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 1 \
  --label_len 96 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_state 16 \
  --d_ff 128 \
  --d_model 128 \
  --train_epochs 30 \
  --patience 7 \
  --e_layers 2 \
  --d_layers 2 \
  --learning_rate 0.00005 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 1 \
  --label_len 96 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --d_state 16 \
  --d_ff 128 \
  --d_model 128 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 50 \
  --patience 10 \
  --e_layers 2 \
  --d_layers 2 \
  --itr 1



