CUDA_VISIBLE_DEVICES=6,7 python script_training.py \
    --model 'LIGHTFUSE_sigmoid' \
    --run_name 'sice' \
    --train_data './dataset_sice_train_patch' \
    --valid_data './dataset_sice_valid' \
    --nChannel 6 \
    --nFeat 32 \
    --loss 'vgg' \
    --epochs 10 \
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
