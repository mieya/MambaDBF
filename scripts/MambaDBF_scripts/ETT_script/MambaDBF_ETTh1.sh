export CUDA_VISIBLE_DEVICES=1

model_name=MambaDBF
d_model=512
d_ff=512
dstate=16
e_layers=1
d_layers=1
seq_len=96
random_seed=2024

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_ff 128 \
  --d_model 128 \
  --d_state 16 \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 1 \
  --patience 5 \
  --train_epochs 30 \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_ff 128 \
  --d_model 128 \
  --d_state 16 \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 2 \
  --patience 5 \
  --train_epochs 30 \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 336 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_ff 128 \
  --d_model 128 \
  --d_state 8 \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 2 \
  --patience 5 \
  --train_epochs 30 \
  --learning_rate 0.0001  \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 720 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_ff 128 \
  --d_model 128 \
  --d_state 8 \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 2 \
  --patience 5 \
  --train_epochs 30 \
  --learning_rate 0.0001 \
  --itr 1


