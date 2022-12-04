import json
import tempfile
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval

from data.coco import COCODataset
from data.stanford_dog import StanfordDogDataset

class StanfordDogAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.img_size = img_size
        self.transform = transform
        self.device = device
        self.map = -1.

        self.testset = testset


        self.dataset = StanfordDogDataset(
            data_dir=data_dir,
            img_size=img_size,
            transform=None,
            image_set='val')


    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            if self.transform is not None:
                x = torch.from_numpy(self.transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(self.device)
            scale = np.array([[img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]]])
            
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                outputs = model(x)
                bboxes, scores, labels = outputs
                bboxes *= scale
            # print('bboxes', bboxes)
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                # label = self.dataset.class_ids[int(labels[i])]
                label = labels[i] + 1
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']
        # print('vaild predict: ', len(data_dict))
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            # if self.testset:
            #     json.dump(data_dict, open('yolo_2017.json', 'w'))
            #     cocoDt = cocoGt.loadRes('yolo_2017.json')
            # else:
            #     _, tmp = tempfile.mkstemp()
            #     json.dump(data_dict, open(tmp, 'w'))
            #     cocoDt = cocoGt.loadRes(tmp)
            # json.dump(data_dict, open('yolo_2017.json', 'w'))
            cocoDt = cocoGt.loadRes(data_dict)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            return ap50_95, ap50
        else:
            return 0, 0

