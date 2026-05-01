# zh
export CUDA_VISIBLE_DEVICES=0

for SEED in 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 
do
    echo "Running seed=${SEED} (zh)"
    python main.py \
        --seed ${SEED} \
        --result_file_name result_seed${SEED} \
        --lang zh \
        --dscgnn_layer_num 2 \
        --gnn_layer_num 3 \
        --loss_w 296 \
        --topk 0.8 \
        --warmup_steps 350 \
        --adam_epsilon 1e-7 \
        --input_files "train_dependent_trf valid_dependent_trf test_dependent_trf"
done


# en
for SEED in 50 51 52 54 55 56 57 58 59 60 61 62 63 64 65 66 
do
    echo "Running seed=${SEED} (en)"
    python main.py \
        --seed ${SEED} \
        --result_file_name result_seed${SEED} \
        --lang en \
        --dscgnn_layer_num 2 \
        --gnn_layer_num 3 \
        --loss_w 296 \
        --topk 0.5 \
        --warmup_steps 400 \
        --adam_epsilon 1e-8 \
        --input_files "train_dependent_trf valid_dependent_trf test_dependent_trf"
done
