cd src

nohup python  run_gsat.py --method macro --c_in raw --cuda 4 --dataset ogbg_molhiv --backbone GIN --r 0.9 --early_stopping 999 &  #
nohup python  run_gsat.py --cuda 0 --dataset Graph-SST2 --backbone GIN --r 0.2 --early_stopping 3 &
nohup python  run_gsat.py --cuda 2 --dataset mnist --backbone GIN --r 0.5 --early_stopping 30 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.5 --backbone GIN --r 0.6 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.7 --backbone GIN --r 0.6 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.9 --backbone GIN --r 0.4 &

nohup python  run_gsat.py --method macro --c_in raw --cuda 4 --dataset ogbg_molhiv --backbone PNA --r 0.7 --early_stopping 999 &  #
nohup python  run_gsat.py --cuda 0 --dataset Graph-SST2 --backbone PNA --r 0.8 --early_stopping 3 &
nohup python  run_gsat.py --cuda 2 --dataset mnist --backbone PNA --r 0.6 --early_stopping 30 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.5 --backbone PNA --r 0.1 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.7 --backbone PNA --r 0.3 &
nohup python  run_gsat.py --cuda 2 --dataset spmotif_0.9 --backbone PNA --r 0.5 &

nohup python  run_gsat.py --method macro --c_in feat --early_stopping 999   --cuda 2 --dataset ogbg_molbace --backbone PNA --r 0.5 &
nohup python  run_gsat.py --method gstopr --c_in feat --early_stopping 999   --cuda 4 --dataset ogbg_molbbbp --backbone PNA --r 0.8 &
nohup python  run_gsat.py --method gstopr --c_in feat --early_stopping 999   --cuda 3 --dataset ogbg_molclintox --backbone PNA --r 0.7 &
nohup python  run_gsat.py --method gstopr --c_in feat --early_stopping 999   --cuda 6 --dataset ogbg_moltox21 --backbone PNA --r 0.7 &
nohup python  run_gsat.py --method gstopr --c_in raw --early_stopping 999   --cuda 0 --dataset ogbg_molsider --backbone PNA --r 0.8 &