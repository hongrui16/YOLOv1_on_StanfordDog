python train.py \
        --cuda \
        -d StanfordDog \
        --batch_size 128 \
        --lr 0.001 \
        --root /home/hongrui/project/dataset \
        --gpu 1 \
        --arch resnet34
        # --max_epoch 150 \
        # --lr_epoch 90 120 \
        # -ms \
        