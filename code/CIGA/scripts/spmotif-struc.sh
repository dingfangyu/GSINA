# spmotif-struc b=0.33
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.33 --r 0.25 --contrast 8 --spu_coe 0 --model 'gcn' --dropout 0.  
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.33 --r 0.25 --contrast 8 --spu_coe 1 --model 'gcn' --dropout 0. 
# spmotif-struc b=0.60
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.60 --r 0.25 --contrast 0.5 --spu_coe 0 --model 'gcn' --dropout 0.  
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.60 --r 0.25 --contrast 4   --spu_coe 2 --model 'gcn' --dropout 0.  
# spmotif-struc b=0.90
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.90 --r 0.25 --contrast 4 --spu_coe 0 --model 'gcn' --dropout 0. 
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.90 --r 0.25 --contrast 4 --spu_coe 2 --model 'gcn' --dropout 0. 



# 
nohup python main.py --device 0 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'spmotif_0.5' --r 0.25 --contrast 8 --spu_coe 1 --model 'gcn' --dropout 0. &
nohup python main.py --device 1 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'spmotif_0.7' --r 0.25 --contrast 4   --spu_coe 2 --model 'gcn' --dropout 0.  & 
nohup python main.py --device 2 -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'spmotif_0.9' --r 0.25 --contrast 4 --spu_coe 2 --model 'gcn' --dropout 0. &
nohup python main.py --device 0 --r 0.8 --num_layers 3  --batch_size 32 --emb_dim 32 --model 'gcn' -c_dim 128 --dataset 'mnist' --seed '[1,2,3,4,5]' --contrast 4 --spu_coe 1 -c_in 'raw'  -c_rep 'feat' &
nohup python main.py --device 2 --r 0.5 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-SST2' --seed '[1,2,3,4,5]' --contrast 4 --spu_coe 1 -c_in 'raw'  -c_rep 'feat' &
nohup python main.py --device 0 --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset ogbg-molhiv --seed '[1,2,3,4,5]' --dropout 0.5 --contrast 16 -c_in 'feat'  -c_rep 'feat' -s_rep 'conv'  --spu_coe 1 &