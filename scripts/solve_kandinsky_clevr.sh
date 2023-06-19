export CUDA_VISIBLE_DEVICES=0
# learning on clevr-hans
python src/train_neumann_grad_scoring.py --dataset-type clevr-hans --dataset clevr-hans2 --num-objects 10 --batch-size 256 --device 0 --epochs 10 --infer-step 2 --trial 6 --n-sample 5 --program-size 1 --min-body-len 6 --max-var 6 --min-body-len 8 --pos-ratio 0.01 --neg-ratio 0.01

# learning on Kandinsky
python src/train_neumann_grad_scoring.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 256 --device 0 --epochs 20 --infer-step 4 --trial 5 --n-sample 5 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 0.01 --neg-ratio 0.01