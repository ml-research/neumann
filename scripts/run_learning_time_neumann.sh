export CUDA_VISIBLE_DEVICES=1

#python src/solve_kandinsky.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.2
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.4
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.6
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.8
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset twopairs --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 1.0

#python src/solve_kandinsky.py --dataset-type kandinsky --dataset closeby --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.2
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset closeby --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.4
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset closeby --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.6
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset closeby --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.8
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset closeby --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 1.0

#python src/solve_kandinsky.py --dataset-type kandinsky --dataset red-triangle --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.2
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset red-triangle --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.4
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset red-triangle --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.6
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset red-triangle --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 0.8
#python src/solve_kandinsky.py --dataset-type kandinsky --dataset red-triangle --num-objects 10 --batch-size 256 --device 0 --epochs 1 --infer-step 2 --n-ratio 1.0

export CUDA_VISIBLE_DEVICES=0
nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.01 --neg-ratio 0.01 > log/neumann_learning_ratio_0.01.out &
export CUDA_VISIBLE_DEVICES=1
nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.05 --neg-ratio 0.05 > log/neumann_learning_ratio_0.05.out &
export CUDA_VISIBLE_DEVICES=2
nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.1 --neg-ratio 0.1 > log/neumann_learning_ratio_0.1.out &
export CUDA_VISIBLE_DEVICES=3
nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.3 --neg-ratio 0.3 > log/neumann_learning_ratio_0.3.out &
export CUDA_VISIBLE_DEVICES=5
nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.5 --neg-ratio 0.5 > log/neumann_learning_ratio_0.5.out &
#export CUDA_VISIBLE_DEVICES=5
#nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 1.0 --neg-ratio 1.0 > log/neumann_learning_ratio_1.0.out &
# 
# export CUDA_VISIBLE_DEVICES=0
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 20 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.01 --neg-ratio 0.01 > log/neumann_learning_ratio_0.01.out &
# export CUDA_VISIBLE_DEVICES=1
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 20 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.05 --neg-ratio 0.05 > log/neumann_learning_ratio_0.05.out &
# export CUDA_VISIBLE_DEVICES=2
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 20 --program-size 1 --max-var 6 --min-body-len 6 --pos-ratio 0.1 --neg-ratio 0.1 > log/neumann_learning_ratio_0.1.out &
# #export CUDA_VISIBLE_DEVICES=3
# #nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 20 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.3 --neg-ratio 0.3 > log/neumann_learning_ratio_0.3.out &
# export CUDA_VISIBLE_DEVICES=4
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 0 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.5 --neg-ratio 0.5 > log/neumann_learning_ratio_0.5.out &
# #export CUDA_VISIBLE_DEVICES=5
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans0 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 4 --n-sample 10 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 1.0 --neg-ratio 1.0 > log/neumann_learning_ratio_1.0.out &

# export CUDA_VISIBLE_DEVICES=0
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.01 --neg-ratio 0.01 > log/neumann_learning_ratio_0.01.out &
# export CUDA_VISIBLE_DEVICES=1
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.05 --neg-ratio 0.05 > log/neumann_learning_ratio_0.05.out &
# export CUDA_VISIBLE_DEVICES=2
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.1 --neg-ratio 0.1 > log/neumann_learning_ratio_0.1.out &
# export CUDA_VISIBLE_DEVICES=3
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.3 --neg-ratio 0.3 > log/neumann_learning_ratio_0.3.out &
# export CUDA_VISIBLE_DEVICES=4
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.5 --neg-ratio 0.5 > log/neumann_learning_ratio_0.5.out &
# export CUDA_VISIBLE_DEVICES=6
# nohup python src/train_kandinsky_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 4 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 1.0 --neg-ratio 1.0 > log/neumann_learning_ratio_1.0.out &