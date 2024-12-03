export CUDA_VISIBLE_DEVICES=0
model_name=MambaDBF

dstate=16
seq_len=192
e_layers=1
d_layers=1
e_fact=1
random_seed=11

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --e_fact $e_fact \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state $dstate \
  --learning_rate 0.0007 \
  --train_epochs 30 \
  --batch_size 8 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 1 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --train_epochs 30 \
  --d_ff 128 \
  --d_state 8 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 4 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 8 \
  --learning_rate 0.0005 \
  --train_epochs 30 \
  --batch_size 8 \
  --itr 1

python -u run.py \
  --task_name ESDWL \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --random_seed $random_seed \
  --e_fact 1 \
  --dconv 1 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 3 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --learning_rate 0.0005 \
  --train_epochs 30 \
  --batch_size 8 \
  --itr 1
