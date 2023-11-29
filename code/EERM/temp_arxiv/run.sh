
# ogb-arxiv
nohup python main.py --method erm --gnn sage --lr 0.005 --dataset ogb-arxiv --device 6 &
python main.py --method eerm --gnn sage --lr 0.005 --K 5 --T 5 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device 2
nohup python main.py --method gstopr --gnn sage --lr 0.005 --dataset ogb-arxiv --device 5 & 
nohup python main.py --method gstopr --noise 0 --gnn sage --lr 0.005 --dataset ogb-arxiv --device 5 & 
