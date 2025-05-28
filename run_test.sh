CUDA_VISIBLE_DEVICES=0 python ./source/script_testing.py \
    --model 'PLAIN_CONV_RES' \
    --run_name 'loss_mse' \
    --test_data './dataset_kalan_test' \
    --input_tonemap 'mu' \
    --label_tonemap 'mu' \
    --epoch 40 \
    --nChannel 3 \
    --nFeat 32
