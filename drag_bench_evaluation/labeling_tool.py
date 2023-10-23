# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import cv2
import numpy as np
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import os
import gradio as gr
import datetime
import pickle
from copy import deepcopy

LENGTH=480 # length of the square area displaying/editing images

def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length), \
        gr.Image.update(value=None, height=length, width=length), \
        [], None, None

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask

# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []

def save_all(category,
             source_image,
             image_with_clicks,
             mask,
             labeler,
             prompt,
             points,
             root_dir='./drag_bench_data'):
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    if not os.path.isdir(os.path.join(root_dir, category)):
        os.mkdir(os.path.join(root_dir, category))

    save_prefix = labeler + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_dir = os.path.join(root_dir, category, save_prefix)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # save images
    Image.fromarray(source_image).save(os.path.join(save_dir, 'original_image.png'))
    Image.fromarray(image_with_clicks).save(os.path.join(save_dir, 'user_drag.png'))

    # save meta data
    meta_data = {
        'prompt' : prompt,
        'points' : points,
        'mask' : mask,
    }
    with open(os.path.join(save_dir, 'meta_data.pkl'), 'wb') as f:
        pickle.dump(meta_data, f)

    return save_prefix + " saved!"

with gr.Blocks() as demo:
    # UI components for editing real images
    with gr.Tab(label="Editing Real Image"):
        mask = gr.State(value=None) # store mask
        selected_points = gr.State([]) # store points
        original_image = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH, width=LENGTH) # for mask painting
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH, width=LENGTH) # for points clicking

        with gr.Row():
            labeler = gr.Textbox(label="Labeler")
            category = gr.Dropdown(value="art_work",
                    label="Image Category",
                    choices=[
                        'art_work',
                        'land_scape',
                        'building_city_view',
                        'building_countryside_view',
                        'animals',
                        'human_head',
                        'human_upper_body',
                        'human_full_body',
                        'interior_design',
                        'other_objects',
                    ]
                )
            prompt = gr.Textbox(label="Prompt")
            save_status = gr.Textbox(label="display saving status")

        with gr.Row():
            undo_button = gr.Button("undo points")
            clear_all_button = gr.Button("clear all")
            save_button = gr.Button("save")

    # event definition
    # event for dragging user-input real image
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas,
        input_image,
        selected_points,
        original_image,
        mask]
    )
    save_button.click(
        save_all,
        [category,
        original_image,
        input_image,
        mask,
        labeler,
        prompt,
        selected_points,],
        [save_status]
    )

demo.queue().launch(share=True, debug=True)
