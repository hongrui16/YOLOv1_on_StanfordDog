# YOLOv1_on_StanfordDog
This repo is forked and edited based on https://github.com/yjh0410/PyTorch_YOLOv1.git

## new features

- add classification model training 
```train_classification_net.py```

- add Stanford Dog dataloader and evaluator
```data/stanford_dog.py```
```evaluator/StanfordDogapi_evaluator.py```

- add dataset arrangement code
```data/dataset_prepare.py```

## train

### classification net training
```train_classification_net.sh```

```
python train_classification_net.py --data dataset/stanfordDogsDataset \
 --logs logs_cls \
 --gpu 0 \
 --evaluate \
 --arch resnet101 \
 --resume logs_cls/resnet101/2022-12-03_23-24-23/model_best.pth.tar
```

### detection net training
```train.sh```

```
python train.py \
        --cuda \
        -d StanfordDog \
        --batch_size 128 \
        --lr 0.001 \
        --root dataset \
        --gpu 2 \
        --arch resnet18 \
        --pretrain
```

## others(push a local repo to another remote repo)
- git remote add [remote_name] [remote_branch_name]
- git remote set-url [remote_name] [remote repo url]
- git push [remote_name] [local_branch]:[remote_branch_name]
- git branch --set-upstream-to=[remote_name]/[remote_branch_name] #set default push repo
