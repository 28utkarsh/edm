CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 generate.py --outdir=generated-samples \
        --network=training-runs/00000-mnist-cond-adm-edm-gpus4-batch512-fp32/network-snapshot-020000.pkl \
        --seeds=0-4999 --batch=64 --steps=40 --n-samples=5000 &

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 generate.py --outdir=generated-samples-seed85 \
        --network=training-runs/00001-mnist-cond-adm-edm-gpus4-batch512-fp32/network-snapshot-020000.pkl \
        --seeds=0-4999 --batch=64 --steps=40 --n-samples=5000 &

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 generate.py --outdir=generated-samples-seed69 \
        --network=training-runs/00002-mnist-cond-adm-edm-gpus4-batch512-fp32/network-snapshot-020000.pkl \
        --seeds=0-4999 --batch=64 --steps=40 --n-samples=5000 &