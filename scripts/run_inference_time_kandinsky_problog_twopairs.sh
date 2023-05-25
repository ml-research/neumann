export CUDA_VISIBLE_DEVICES=13

python src/solve_kandinsky_problog.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.2
python src/solve_kandinsky_problog.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.4
python src/solve_kandinsky_problog.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.6
python src/solve_kandinsky_problog.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.8
python src/solve_kandinsky_problog.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 1.0