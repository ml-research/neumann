#export CUDA_VISIBLE_DEVICES=10
#nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 0.01 > log/ilasp_learning_ratio_0.01.out &
export CUDA_VISIBLE_DEVICES=11
nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 0.05 > log/ilasp_learning_ratio_0.05.out &
export CUDA_VISIBLE_DEVICES=12
nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 0.1 > log/ilasp_learning_ratio_0.1.out &
export CUDA_VISIBLE_DEVICES=13
nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 0.3 > log/ilasp_learning_ratio_0.3.out &
export CUDA_VISIBLE_DEVICES=14
nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 0.5 > log/ilasp_learning_ratio_0.5.out &
export CUDA_VISIBLE_DEVICES=15
nohup python src/solve_kandinsky_ilasp.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 200 --device 0 --n-ratio 1.0 > log/ilasp_learning_ratio_1.0.out &


# memo
python src/solve_kandinsky_ilasp.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 200 --device 0 --n-ratio 0.01