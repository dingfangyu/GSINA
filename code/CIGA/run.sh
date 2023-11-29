
# Graph-SST5
nohup python main.py --device 0 --ginv_opt macro --r 0.5 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-SST5' --seed '[1,2,3,4,5]'  -c_in 'raw'  -c_rep 'feat' --pretrain 0 --early_stopping 3 &

# Graph-Twitter
nohup python main.py --device 2 --ginv_opt macro --r 0.6 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-Twitter' --seed '[1,2,3,4,5]'  -c_in 'feat'  -c_rep 'feat'  --pretrain 5 --early_stopping 5 &


# spmotif-struc b=0.33
nohup python main.py --device 1 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.33 --r 0.25 --model 'gcn' --dropout 0. & 

# spmotif-struc b=0.60
nohup python main.py --device 2 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.6 --r 0.25 --model 'gcn' --dropout 0. & 

# spmotif-struc b=0.90
nohup python main.py --device 3 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.9 --r 0.25 --model 'gcn' --dropout 0. &


# mspmotif-struc b=0.33
nohup python main.py --device 1 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'mSPMotif' --bias 0.33 --r 0.25 --model 'gcn' --dropout 0.  &

# mspmotif-struc b=0.60
nohup python main.py --device 2 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'mSPMotif' --bias 0.6 --r 0.25 --model 'gcn' --dropout 0.  &

# mspmotif-struc b=0.90
nohup python main.py --device 3 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'mSPMotif' --bias 0.9 --r 0.25 --model 'gcn' --dropout 0. &


# TU
nohup python main.py --device 6 --ginv_opt macro -c_in 'feat' -c_rep 'feat' --num_layers 3 --dataset 'nci1' --r 0.6 --model 'gcn' --pretrain 30 --early_stopping 10 --dropout 0.3 --eval_metric 'mat'  & 
nohup python main.py --device 2 --ginv_opt macro  -c_in 'feat' -c_rep 'feat' --num_layers 3 --dataset 'nci109' --r 0.7 --model 'gcn' --pretrain 30 --early_stopping 10 --dropout 0.3 --eval_metric 'mat'  &
nohup python main.py --device 5  -c_in 'raw' -c_rep 'rep' --num_layers 3 --dataset 'proteins' --r 0.3 --pretrain 30 --early_stopping 10 --model 'gin' --pooling 'max' --dropout 0.3 --eval_metric 'mat'  & 
nohup python main.py --device 6  -c_in 'raw' -c_rep 'rep' --num_layers 3 --dataset 'dd' --r 0.3 --model 'gcn' --pretrain 30 --early_stopping 10 --dropout 0.3 --eval_metric 'mat'  &

# DrugOOD
nohup python main.py --device 0 --ginv_opt macro  --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_assay' --seed '[1,2,3,4,5]' --dropout 0.5  -c_in 'raw'  -c_rep 'feat'  --pretrain 20 --early_stopping 20 &
nohup python main.py --device 5 --ginv_opt macro   --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5  -c_in 'feat'  -c_rep 'feat' -s_rep 'conv'  --pretrain 20 --early_stopping 20 &
nohup python main.py --device 5 --ginv_opt macro   --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_size' --seed '[1,2,3,4,5]' --dropout 0.1  -c_in 'feat'  -c_rep 'feat'  --pretrain 20 --early_stopping 10 &