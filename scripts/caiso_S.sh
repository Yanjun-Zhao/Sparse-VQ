python -u run_longExp.py       --random_seed 2023       --is_training 1       --root_path  ./dataset/caiso       --data_path caiso.csv       --model_id caiso       --model SVQ       --data custom       --features S       --seq_len 512       --pred_len 96       --enc_in 1       --e_layers 3       --n_heads 16       --d_model 128       --d_ff 256       --dropout 0.2      --fc_dropout 0.2      --head_dropout 0      --patch_len 64      --stride 32   --sout 0   --des 'Exp'       --train_epochs 100      --patience 5      --pct_start 0.4    --itr 1 --batch_size 64 --learning_rate 0.0001 --svq 1 --wFFN 0 >logs/SVQ_S_caiso_512_96.log


python -u run_longExp.py       --random_seed 2023       --is_training 1       --root_path  ./dataset/caiso       --data_path caiso.csv       --model_id caiso       --model SVQ       --data custom       --features S       --seq_len 512       --pred_len 192      --enc_in 1       --e_layers 3       --n_heads 16       --d_model 128       --d_ff 256       --dropout 0.2      --fc_dropout 0.2      --head_dropout 0      --patch_len 64      --stride 32   --sout 0   --des 'Exp'       --train_epochs 100      --patience 5      --pct_start 0.4    --itr 1 --batch_size 64 --learning_rate 0.0001 --svq 1 --wFFN 0  >logs/SVQ_S_caiso_512_192.log


python -u run_longExp.py       --random_seed 2023       --is_training 1       --root_path  ./dataset/caiso       --data_path caiso.csv       --model_id caiso       --model SVQ       --data custom       --features S       --seq_len 512       --pred_len 336       --enc_in 1       --e_layers 3       --n_heads 16       --d_model 128       --d_ff 256       --dropout 0.2      --fc_dropout 0.2      --head_dropout 0      --patch_len 64      --stride 32    --sout 0  --des 'Exp'       --train_epochs 100      --patience 5      --pct_start 0.4    --itr 1 --batch_size 64 --learning_rate 0.0001  --svq 1 --wFFN 0  >logs/SVQ_S_caiso_512_336.log


python -u run_longExp.py       --random_seed 2023       --is_training 1       --root_path  ./dataset/caiso       --data_path caiso.csv       --model_id caiso       --model SVQ       --data custom       --features S       --seq_len 512       --pred_len 720       --enc_in 1       --e_layers 3       --n_heads 16       --d_model 128       --d_ff 256       --dropout 0.2      --fc_dropout 0.2      --head_dropout 0      --patch_len 64      --stride 32   --sout 0   --des 'Exp'       --train_epochs 100      --patience 5      --pct_start 0.4    --itr 1 --batch_size 64 --learning_rate 0.0001  --svq 1 --wFFN 0  >logs/SVQ_S_caiso_512_720.log








