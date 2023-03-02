export CUDA_VISIBLE_DEVICES=2
nohup python src/train.py -dt vilp -ds reverse -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 10 -mv 4 -s 0 > out/reverse_0.out &
export CUDA_VISIBLE_DEVICES=3
nohup python src/train.py -dt vilp -ds reverse -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 10 -mv 4 -s 1 > out/reverse_1.out &
export CUDA_VISIBLE_DEVICES=4
nohup python src/train.py -dt vilp -ds reverse -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 10 -mv 4 -s 2 > out/reverse_2.out &
#export CUDA_VISIBLE_DEVICES=5
#nohup python src/train.py -dt vilp -ds reverse -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 10 -mv 4 -s 3 > out/reverse_3.out &
#export CUDA_VISIBLE_DEVICES=6
#nohup python src/train.py -dt vilp -ds reverse -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 10 -mv 4 -s 4 > out/reverse_4.out &
