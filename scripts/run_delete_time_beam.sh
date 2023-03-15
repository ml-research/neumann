export CUDA_VISIBLE_DEVICES=3
python src/train_time_beam.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 0 > out/delete_beam_0.out 
python src/train_time_beam.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 1 > out/delete_beam_1.out
python src/train_time_beam.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 2 > out/delete_beam_2.out
python src/train_time_beam.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 3 > out/delete_beam_3.out
python src/train_time_beam.py -dt vilp -ds delete -d 0 -ps 2 -bs 4 -tr 5 -is 5 -thd 3 -ns 5 -mv 4 -pd 2 -pr 0.5 -e 200  -s 4 > out/delete_beam_4.out
