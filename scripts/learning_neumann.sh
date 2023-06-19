export CUDA_VISIBLE_DEVICES=0
# learning on clevr-hans
python src/train_neumann_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.01 --neg-ratio 0.01

# learning on Kandinsky
python src/train_neumann_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 5 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.01 --neg-ratio 0.01

# learning clevr-list
python src/train_neumann_grad_scoring.py --dataset-type vilp --dataset delete --num-objects 3 --batch-size 4 --device 0 --epochs 5 --infer-step 6 --trial 4 --n-sample 10 --program-size 2  --max-var 4 --min-body-len 0 --pos-ratio 0.5 --neg-ratio 1.0 --lr 1e-2

# solving behind-the-scenes
 python src/solve_behind_the_scenes_time.py  --device 0 --batch-size 128 --dataset delete