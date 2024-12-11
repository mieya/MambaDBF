export CUDA_VISIBLE_DEVICES=1

model_name=MY
d_model=256
d_ff=256
dstate=16
seq_len=192
e_fact=2
random_seed=11

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 1 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 64 \
  --train_epochs 40 \
  --patience 5 \
  --d_model 256 \
  --d_ff 256 \
  --d_state 32\
  --des 'Exp' \
  --dconv 1 \
  --learning_rate 0.0001 \
  --itr 1



python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 2 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 7 \
  --d_model 128 \
  --d_ff 128 \
  --d_state 8 \
  --dconv 1 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 1 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 64 \
  --train_epochs 30 \
  --patience 5 \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --des 'Exp' \
  --dconv 1 \
  --learning_rate 0.0001 \
  --itr 1 


python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_layers 1 \
  --d_layers 1 \
  --e_fact 2 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 64 \
  --train_epochs 30 \
  --patience 5 \
  --d_model 128 \
  --d_ff 128 \
  --d_state 8 \
  --dconv 1 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --itr 1
