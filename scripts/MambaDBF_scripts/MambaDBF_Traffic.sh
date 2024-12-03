export CUDA_VISIBLE_DEVICES=1

model_name=MambaDBF
dstate=16
lr=0.0001 
d_ff=256
d_model=128
seq_len=192
random_seed=3047

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 3 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 8 \
  --d_ff 128 \
  --d_model 128 \
  --d_state 32 \
  --e_fact 4 \
  --learning_rate $lr \
  --train_epochs 30 \
  --patience 5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 2 \
  --label_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 8 \
  --d_ff 512 \
  --d_model 256 \
  --d_state 16 \
  --learning_rate $lr \
  --des 'Exp' \
  --train_epochs 30 \
  --patience 5 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 2 \
  --label_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 8 \
  --d_ff 512 \
  --d_model 256 \
  --d_state 16 \
  --learning_rate $lr \
  --des 'Exp' \
  --train_epochs 30 \
  --patience 5 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 2 \
  --label_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 8 \
  --d_ff 512 \
  --d_model 256 \
  --d_state 16 \
  --learning_rate $lr \
  --des 'Exp' \
  --train_epochs 30 \
  --patience 5 \
  --itr 1