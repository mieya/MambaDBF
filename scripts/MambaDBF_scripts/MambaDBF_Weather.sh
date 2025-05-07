export CUDA_VISIBLE_DEVICES=0

model_name=MambaDBF

dstate=32
expand=4 
lr=0.0005
e_layers=3
d_layers=3
seq_len=96
random_seed=2024

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 128\
  --e_fact 2 \
  --d_state 8 \
  --learning_rate 0.0001 \
  --e_layers 1 \
  --d_layers 1 \
  --train_epochs 30 \
  --patience 4 \
  --des 'Exp' \
  --itr 1 


python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 192 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 128\
  --d_state 16 \
  --e_fact 2 \
  --learning_rate 0.0001 \
  --e_layers 1 \
  --d_layers 1 \
  --train_epochs 30 \
  --patience 5 \
  --des 'Exp' \
  --itr 1 

  python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 128\
  --d_state 16 \
  --e_fact 2 \
  --learning_rate 0.0001 \
  --e_layers 1 \
  --d_layers 1 \
  --train_epochs 30 \
  --patience 5 \
  --des 'Exp' \
  --itr 1  

  python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 128\
  --d_state 16 \
  --e_fact 2 \
  --learning_rate 0.0001 \
  --e_layers 1 \
  --d_layers 1 \
  --train_epochs 30 \
  --patience 5 \
  --des 'Exp' \
  --itr 1  

