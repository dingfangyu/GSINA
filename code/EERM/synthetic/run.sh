# cora
python main.py --method eerm --dataset cora --gnn_gen gcn --gnn gcn --lr 0.005 --K 10 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 5 --device 5
nohup python main.py --method erm --dataset cora --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 5 &
nohup python main.py --method gstopr --dataset cora --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 5 &
nohup python main.py --method gstopr --noise 0 --dataset cora --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 5 &


# amazon-photo
nohup python main.py --method eerm --dataset amazon-photo --gnn_gen gcn --gnn gcn --lr 0.01 --K 5 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 5 --device 5 &
nohup python main.py --method erm --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 6 &
nohup python main.py --method gstopr --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 6 &
nohup python main.py --method gstopr --noise 0 --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 5 --lr 0.01 --device 6 &
