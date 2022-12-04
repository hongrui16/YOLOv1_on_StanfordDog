##train
# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 2 \
#  --arch resnet18 \
#  --pretrained

python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
 --logs logs_cls \
 --gpu 3 \
 --arch resnet34 \
 --pretrained

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 1 \
#  --arch resnet50 \
#  --pretrained

##eval
# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --resume logs_cls/resnet18/2022-12-03_18-20-15/model_best.pth.tar
