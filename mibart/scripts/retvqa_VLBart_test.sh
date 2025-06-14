# The name of experiment
name=VLBart

output=/home/monu_harsh/Harshwardhan/mi_bart/Results/$name

PYTHONPATH=$PYTHONPATH:./home/monu_harsh/Harshwardhan/mi_bart/mi_bart/src \
python3 -m torch.distributed.launch \
    --nproc_per_node=$1 \
    /home/monu_harsh/Harshwardhan/mi_bart/mi_bart/src/retvqa_only_val.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load /home/monu_harsh/Harshwardhan/mi_bart/pthFiles/VLBart/BEST \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 360 \
        --img_only \
        # --use_qcat
