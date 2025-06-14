# The name of experiment
name=VLBart

output=/home/monu_harsh/new/mi_bart/mi_bart/snap/retvqa/$name

PYTHONPATH=$PYTHONPATH:/home/monu_harsh/new/mi_bart/mi_bart/src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    /home/monu_harsh/new/mi_bart/mi_bart/src/retvqa.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-3 \
        --epochs 100 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load  /home/monu_harsh/new/mi_bart/mi_bart/snap/retvqa/VLBart/BEST \
        --num_beams 5 \
        --batch_size 200 \
        --valid_batch_size 200 \