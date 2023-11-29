
# Twitch-e
python main.py --dataset twitch-e --method erm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 2
python main.py --dataset twitch-e --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 2
nohup python main.py --dataset twitch-e --method gstopr --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 5 &
nohup python main.py --dataset twitch-e --method gstopr --noise 0 --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 5 &


# fb-100
python main.py --dataset fb100 --train 2 --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 0

nohup python main.py --dataset fb100 --train 1 --method gstopr --lr 0.001 --device 6 &
nohup python main.py --dataset fb100 --train 2 --method gstopr --lr 0.001 --device 7 &
nohup python main.py --dataset fb100 --train 3 --method gstopr --lr 0.001 --device 8 &

nohup python main.py --dataset fb100 --train 1 --method gstopr --noise 0 --lr 0.001 --device 6 &
nohup python main.py --dataset fb100 --train 2 --method gstopr --noise 0 --lr 0.001 --device 7 &
nohup python main.py --dataset fb100 --train 3 --method gstopr --noise 0 --lr 0.001 --device 8 &
