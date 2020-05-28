if [ $1 = "1" ]
then
    python postrain.py --num-gpus 4  --config-file ../configs/fast_autoaug_postrain.yaml --resume
elif [ $1 = "2" ]
then
    python postrain.py --num-gpus 4  --config-file ../configs/fast_autoaug_postrain.yaml --skip-pretrain
else
    python postrain.py --num-gpus 4  --config-file ../configs/fast_autoaug_postrain.yaml
fi

