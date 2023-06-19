export CUDA_VISIBLE_DEVICES=0

# learning clevr-list
python src/train_neumann_grad_scoring.py --dataset-type vilp --dataset delete --num-objects 3 --batch-size 4 --device 0 --epochs 5 --infer-step 6 --trial 4 --n-sample 10 --program-size 2  --max-var 4 --min-body-len 0 --pos-ratio 0.5 --neg-ratio 1.0 --lr 1e-2