export CUDA_VISIBLE_DEVICES=4
# python src/train_time.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 0 > out/delete_0.out 
# python src/train_time.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 1 > out/delete_1.out
python src/train_time.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 2 > out/delete_2.out
python src/train_time.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 3 > out/delete_3.out
python src/train_time.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 4 > out/delete_4.out
