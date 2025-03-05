CUDA_VISIBLE_DEVICES=3 python script_testing.py \
    --model "LIGHTFUSE_sigmoid" \
    --run_name 'kalan_tm_in_label' \
    --epoch 20 \
    --test_data './dataset_kalan_test' \
    --input_tonemap 'mu' \
    --label_tonemap 'mu' \
    --nChannel 6 \
    --nFeat 32
