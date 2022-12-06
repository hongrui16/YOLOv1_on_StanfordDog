
python eval.py \
        --cuda \
        -d StanfordDog \
        --root /home/hongrui/project/dataset \
        --gpu 1 \
        --arch resnet18 \
        --resume logs_det/StanfordDog/resnet18/2022-12-04_18-20-10/weight/yolo_epoch_71_56.9.pth \
        --inference

# python eval.py \
#         --cuda \
#         -d StanfordDog \
#         --root /home/hongrui/project/dataset \
#         --gpu 1 \
#         --arch resnet18 \
#         --resume logs_det/StanfordDog/resnet18/2022-12-05_11-42-05/weight/yolo_epoch_61_15.1.pth

python eval.py \
        --cuda \
        -d StanfordDog \
        --root /home/hongrui/project/dataset \
        --gpu 1 \
        --arch resnet18 \
        --input_size 384 \
        --stride 64 \
        --resume logs_det/StanfordDog/resnet18/2022-12-05_20-13-37/weight/yolo_epoch_66_55.6.pth \
        --inference

# python eval.py \
#         --cuda \
#         -d StanfordDog \
#         --root /home/hongrui/project/dataset \
#         --gpu 1 \
#         --arch resnet34 \
#         --resume logs_det/StanfordDog/resnet34/2022-12-04_18-39-34/weight/yolo_epoch_75_61.1.pth

# python eval.py \
#         --cuda \
#         -d StanfordDog \
#         --root /home/hongrui/project/dataset \
#         --gpu 1 \
#         --arch resnet34 \
#         --resume logs_det/StanfordDog/resnet34/2022-12-05_11-42-23/weight/yolo_epoch_51_14.5.pth

# python full_eval.py \
#         --cuda \
#         -d StanfordDog \
#         --root /home/hongrui/project/dataset \
#         --gpu 1 \
#         --arch resnet34 \
#         --input_size 384 \
#         --stride 64 \
#         --resume logs_det/StanfordDog/resnet34/2022-12-05_20-15-28/weight/yolo_epoch_75_57.1.pth
