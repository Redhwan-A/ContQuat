import os

os.system("pip install git+https://github.com/elliottzheng/face-detection.git@master")
os.system("git clone https://github.com/thohemp/6DRepNet")

import sys

sys.path.append("6DRepNet")

import numpy as np
import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from face_detection import RetinaFace
from model import SixDRepNet
import utils
import cv2
from PIL import Image

snapshot_path = hf_hub_download(repo_id="osanseviero/6DRepNet_300W_LP_AFLW2000", filename="model.pth")

model = SixDRepNet(backbone_name='RepVGG-B1g2',
                   backbone_file='',
                   deploy=True,
                   pretrained=False)

detector = RetinaFace(0)
saved_state_dict = torch.load(os.path.join(
    snapshot_path), map_location='cpu')

if 'model_state_dict' in saved_state_dict:
    model.load_state_dict(saved_state_dict['model_state_dict'])
else:
    model.load_state_dict(saved_state_dict)
model.cuda(0)
model.eval()


def predict(frame):
    faces = detector(frame)
    for box, landmarks, score in faces:
        # Print the location of each face in this image
        if score < .95:
            continue
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)

        x_min = max(0, x_min - int(0.2 * bbox_height))
        y_min = max(0, y_min - int(0.2 * bbox_width))
        x_max = x_max + int(0.2 * bbox_height)
        y_max = y_max + int(0.2 * bbox_width)

        img = frame[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (244, 244)) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = torch.Tensor(img).cuda(0)
        img = img.unsqueeze(0)
        R_pred = model(img)
        euler = utils.compute_euler_angles_from_rotation_matrices(
            R_pred) * 180 / np.pi
        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()
        return utils.plot_pose_cube(frame, y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5 * (x_max - x_min)),
                                    y_min + int(.5 * (y_max - y_min)), size=bbox_width)


title = "6D Rotation Representation for Unconstrained Head Pose Estimation"
description = "Gradio demo for 6DRepNet. To use it, simply click the camera picture. Read more at the links below."
article = "<div style='text-align: center;'><a href='https://github.com/thohemp/6DRepNet' target='_blank'>Github Repo</a> | <a href='https://arxiv.org/abs/2202.12555' target='_blank'>Paper</a></div>"

image_flip_css = """
.input-image .image-preview  img{
    -webkit-transform: scaleX(-1);
    transform: scaleX(-1) !important;
}
.output-image img {
    -webkit-transform: scaleX(-1);
    transform: scaleX(-1) !important;
}
"""

iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(label="Input Image", source="webcam"),
    outputs='image',
    live=True,
    title=title,
    description=description,
    article=article,
    css=image_flip_css
)

iface.launch()