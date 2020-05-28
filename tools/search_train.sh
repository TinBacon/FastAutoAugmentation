#export CUDA_VISIBLE_DEVICES=2
if [ $1 = "1" ]
then
    #python search_train.py --num-gpus 4  --config-file ../configs/fast_autoaug.yaml --resume
    python search_train.py --num-gpus 4  --config-file ../configs/fast_autoaug.yaml --skip-pretrain
else
    python search_train.py --num-gpus 4  --config-file ../configs/fast_autoaug.yaml
fi
