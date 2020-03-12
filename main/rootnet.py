import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import math

from dataset import generate_patch_image
from config import cfg as rootnet_cfg
from base import Tester as rootnet_Tester
from utils.pose_utils import pixel2cam

def get_input(image, person_boxes):
    person_images = np.zeros((len(person_boxes), 3, rootnet_cfg.input_shape[0], rootnet_cfg.input_shape[1]))
    k_values = np.zeros((len(person_boxes), 1))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)]
    )

    for i, box in enumerate(person_boxes):
        patch_image, _ = generate_patch_image(image, box, False, 0)
        person_images[i] = transform(patch_image)
        #k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32) in dataset.py
        k_values[i] = np.array(
            [math.sqrt(rootnet_cfg.bbox_real[0] * rootnet_cfg.bbox_real[1] * (image.shape[1]/2) * (image.shape[0]/2) / (box[3] * box[2]))]).astype(
            np.float32)

    person_images = torch.Tensor(person_images)
    k_values = torch.Tensor(k_values)

    return person_images, k_values

def set_rootnet_config():
    rootnet_cfg.set_args(gpu_ids='0')
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

def get_rootnet_model():
    #rootNet model: snapshot_18.pth.tar => idx = 18
    rootnet_tester = rootnet_Tester(18)
    rootnet_tester._make_model()
    return rootnet_tester

def get_root(raw_image, person_boxes, rootnet_model, person_images, k_values):
    #assume f = [960,540] width/2, height/2
    f = [960,540]
    with torch.no_grad():
        rootnet_preds = rootnet_model.model(person_images, k_values)
        rootnet_preds = rootnet_preds.cpu().numpy()

    for i, box in enumerate(person_boxes):
        rootnet_pred = rootnet_preds[i]
        rootnet_pred[0] = rootnet_pred[0] / rootnet_cfg.output_shape[1] * box[2] + box[0]
        rootnet_pred[1] = rootnet_pred[1] / rootnet_cfg.output_shape[0] * box[3] + box[1]
        # pixel2cam converts (x_img, y_img, z_cam) to (x_cam, y_cam, z_cam)
        # rootnet_pred[0], rootnet_pred[1], rootnet_pred[2] = pixel2cam(rootnet_pred, custom_cfg.f, np.array([raw_image.shape[1]/2, raw_image.shape[0]/2]))
    return rootnet_preds