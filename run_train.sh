CUDA_VISIBLE_DEVICES=0,1 python script_training.py \
    --model 'LIGHTFUSE_3exp' \
    --run_name 'kalan_f32' \
    --train_data './dataset_kalan_train_patch_dior' \
    --valid_data './dataset_kalan_test' \
    --input_tonemap 'mu' \
    --label_tonemap 'mu' \
    --nChannel 9 \
    --nFeat 32 \
    --loss 'vgg' \
    --epochs 40 \
    --batchsize 128
#CUDA_VISIBLE_DEVICES=6,7 python script_training.py \
#    --model 'LIGHTFUSE_sigmoid' \
#    --run_name 'kalan_notm' \
#    --train_data './dataset_kalan_train_patch' \
#    --valid_data './dataset_kalan_test' \
#    --nChannel 6 \
#    --nFeat 32 \
#    --loss 'vgg' \
#    --epochs 20 \
#    --batchsize 128
