from .yolo import myYOLO


def build_yolo(args, device, train_size, num_classes=20, trainable=False, pretrain = True):
    backbone_arch = args.arch
    stride = args.stride
    model = myYOLO(device=device,
                   input_size=train_size, 
                   num_classes=num_classes,
                   trainable=trainable, backbone_arch = backbone_arch, pretrain = pretrain, stride=stride)

    return model
