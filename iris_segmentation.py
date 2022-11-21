import os
import sys
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image

sys.path.append('./detection')
import utils
from engine import train_one_epoch, evaluate

with open("./env_var.json", "r") as env_var_json:
    env_var = json.load(env_var_json)
    image_train_dir_path = env_var['train_image_dir_path']
    mask_train_dir_path = env_var['train_mask_dir_path']
    train_data_num = env_var['train_data_num']
    print('### image_train_dir_path : ', image_train_dir_path)
    print('### mask_train_dir_path : ', mask_train_dir_path)
    print('### train_data_num : ', train_data_num)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, transforms):
        self.imge_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(image_path)))
        self.masks = list(sorted(os.listdir(mask_path)))

        # If there is no segmentation mask, then remove the data
        rm_name_list = ['C201_S1_I6_R', 'C487_S1_I14_L', 'C247_S2_I13_L', 'C205_S1_I12_R', 'C495_S1_I2_L', 'C205_S1_I6_R']
        rm_list = []
        for rm_name in rm_name_list:
          rm_list.extend([rm_name + '.jpg', rm_name + '.png'])
        
        self.imgs = [i for i in self.imgs if i not in rm_list]
        self.masks = [i for i in self.masks if i not in rm_list]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.image_path, self.imgs[idx])
        mask_path = os.path.join(self.mask_path, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        # Try to find removing data.
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except:
          print(self.imgs[idx], self.masks[idx], image_id, boxes)          
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 4

    train = True
    if train:
        # load dataset
        dataset = TrainDataset(image_train_dir_path, mask_train_dir_path, get_transform(train=True))
        dataset_test = TrainDataset(image_train_dir_path, mask_train_dir_path, get_transform(train=False))
        
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:8500])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[8500:])
        
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)
        
        # get the model using our helper function
        model = get_model_instance_segmentation(num_classes)
        # move model to the right device
        model.to(device)
        
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
        # let's train it for 10 epochs
        num_epochs = 10
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)