CUDA_VISIBLE_DEVICES=0 python script_testing.py \
    --model 'LIGHTFUSE_add_last_depthwise' \
    --run_name 'sice' \
    --epoch 10 \
    --test_data './dataset_sice_valid' \
    --nChannel 6 \
    --nFeat 32
