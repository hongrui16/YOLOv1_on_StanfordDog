##train
# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 2 \
#  --arch resnet18 \
#  --pretrained

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 3 \
#  --arch resnet34 \
#  --pretrained

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 1 \
#  --arch resnet50 \
#  --pretrained

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 1 \
#  --arch resnet101 \
#  --pretrained

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --arch resnet101 \
#  --pretrained

##eval
# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --resume logs_cls/resnet18/2022-12-03_18-20-15/model_best.pth.tar

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --resume logs_cls/resnet18/2022-12-03_20-55-51/model_best.pth.tar

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --arch resnet34 \
#  --resume logs_cls/resnet34/2022-12-03_20-48-23/model_best.pth.tar

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --arch resnet34 \
#  --resume logs_cls/resnet34/2022-12-03_20-56-22/model_best.pth.tar

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --arch resnet50 \
#  --resume logs_cls/resnet50/2022-12-03_20-51-46/model_best.pth.tar

# python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
#  --logs logs_cls \
#  --gpu 0 \
#  --evaluate \
#  --arch resnet50 \
#  --resume logs_cls/resnet50/2022-12-03_22-10-34/model_best.pth.tar

python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
 --logs logs_cls \
 --gpu 0 \
 --evaluate \
 --arch resnet101 \
 --resume logs_cls/resnet101/2022-12-03_23-24-23/model_best.pth.tar

python train_classification_net.py --data /home/hongrui/project/dataset/stanfordDogsDataset \
 --logs logs_cls \
 --gpu 0 \
 --evaluate \
 --arch resnet101 \
 --resume logs_cls/resnet101/2022-12-03_23-25-06/model_best.pth.tar