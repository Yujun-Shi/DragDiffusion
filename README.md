<p align="center">
  <h1 align="center">DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing</h1>
  <p align="center">
    <a href="https://yujun-shi.github.io/"><strong>Yujun Shi</strong></a>
    &nbsp;&nbsp;
    <strong>Chuhui Xue</strong>
    &nbsp;&nbsp;
    <strong>Jiachun Pan</strong>
    &nbsp;&nbsp;
    <strong>Wenqing Zhang</strong>
    &nbsp;&nbsp;
    <a href="https://vyftan.github.io/"><strong>Vincent Y. F. Tan</strong></a>
    &nbsp;&nbsp;
    <a href="https://songbai.site/"><strong>Song Bai</strong></a>
  </p>
  <br>
  <div align="center">
    <img src="./release-doc/asset/counterfeit-1.png", width="700">
    <img src="./release-doc/asset/counterfeit-2.png", width="700">
    <img src="./release-doc/asset/majix_realistic.png", width="700">
  </div>
  <div align="center">
    <img src="./release-doc/asset/github_video.gif", width="700">
  </div>
  <p align="center">
    <a href="https://arxiv.org/abs/2306.14435"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2306.14435-b31b1b.svg"></a>
    <a href="https://yujun-shi.github.io/projects/dragdiffusion.html"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://twitter.com/YujunPeiyangShi"><img alt='Twitter' src="https://img.shields.io/twitter/follow/YujunPeiyangShi?label=%40YujunPeiyangShi"></a>
  </p>
  <br>
</p>

## Disclaimer
This is a research project, NOT a commercial product.

## News and Update
* [Sept 3rd] v0.1.0 Release.
  * Enable **Dragging Diffusion-Generated Images.**
  * Introducing a new guidance mechanism that **greatly improve quality of dragging results.** (Inspired by [MasaCtrl](https://ljzycmd.github.io/projects/MasaCtrl/))
  * Enable Dragging Images with arbitrary aspect ratio
  * Adding support for DPM++Solver (Generated Images)
* [July 18th] v0.0.1 Release.
  * Integrate LoRA training into the User Interface. **No need to use training script and everything can be conveniently done in UI!**
  * Optimize User Interface layout.
  * Enable using better VAE for eyes and faces (See [this](https://stable-diffusion-art.com/how-to-use-vae/))
* [July 8th] v0.0.0 Release.
  * Implement Basic function of DragDiffusion

## Installation

It is recommended to run our code on a Nvidia GPU with a linux system. We have not yet tested on other configurations. Currently, it requires around 14 GB GPU memory to run our method. We will continue to optimize memory efficiency

To install the required libraries, simply run the following command:
```
conda env create -f environment.yaml
conda activate dragdiff
```

## Run DragDiffusion
To start with, in command line, run the following to start the gradio user interface:
```
python3 drag_ui.py
```

You may check our [GIF above](https://github.com/Yujun-Shi/DragDiffusion/blob/main/release-doc/asset/github_video.gif) that demonstrate the usage of UI in a step-by-step manner.

Basically, it consists of the following steps:

#### Step 1: train a LoRA
1) Drop our input image into the left-most box.
2) Input a prompt describing the image in the "prompt" field
3) Click the "Train LoRA" button to train a LoRA given the input image

#### Step 2: do "drag" editing
1) Draw a mask in the left-most box to specify the editable areas.
2) Click handle and target points in the middle box. Also, you may reset all points by clicking "Undo point".
3) Click the "Run" button to run our algorithm. Edited results will be displayed in the right-most box.


## Explanation for parameters in the user interface:
#### General Parameters
|Parameter|Explanation|
|-----|------|
|prompt|The prompt describing the user input image (This will be used to train the LoRA and conduct "drag" editing).|
|lora_path|The directory where the trained LoRA will be saved.|


#### Algorithm Parameters
These parameters are collapsed by default as we normally do not have to tune them. Here are the explanations:
* Base Model Config

|Parameter|Explanation|
|-----|------|
|Diffusion Model Path|The path to the diffusion models. By default we are using "runwayml/stable-diffusion-v1-5". We will add support for more models in the future.|
|VAE Choice|The Choice of VAE. Now there are two choices, one is "default", which will use the original VAE. Another choice is "stabilityai/sd-vae-ft-mse", which can improve results on images with human eyes and faces (see [explanation](https://stable-diffusion-art.com/how-to-use-vae/))|

* Drag Parameters

|Parameter|Explanation|
|-----|------|
|n_pix_step|Maximum number of steps of motion supervision. **Increase this if handle points have not been "dragged" to desired position.**|
|lam|The regularization coefficient controlling unmasked region stays unchanged. Increase this value if the unmasked region has changed more than what was desired (do not have to tune in most cases).|
|n_actual_inference_step|Number of DDIM inversion steps performed (do not have to tune in most cases).|

* LoRA Parameters

|Parameter|Explanation|
|-----|------|
|LoRA training steps|Number of LoRA training steps (do not have to tune in most cases).|
|LoRA learning rate|Learning rate of LoRA (do not have to tune in most cases)|
|LoRA rank|Rank of the LoRA (do not have to tune in most cases).|


## License
Code related to the DragDiffusion algorithm is under Apache 2.0 license.


## BibTeX
```bibtex
@article{shi2023dragdiffusion,
  title={DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing},
  author={Shi, Yujun and Xue, Chuhui and Pan, Jiachun and Zhang, Wenqing and Tan, Vincent YF and Bai, Song},
  journal={arXiv preprint arXiv:2306.14435},
  year={2023}
}
```

## Contact
For any questions on this project, please contact [Yujun](https://yujun-shi.github.io/) (shi.yujun@u.nus.edu)

## Acknowledgement
This work is inspired by the amazing [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/). The lora training code is modified from an [example](https://github.com/huggingface/diffusers/blob/v0.17.1/examples/dreambooth/train_dreambooth_lora.py) of diffusers. Image samples are collected from [unsplash](https://unsplash.com/), [pexels](https://www.pexels.com/zh-cn/), [pixabay](https://pixabay.com/). Finally, a huge shout-out to all the amazing open source diffusion models and libraries.

## Related Links
* [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
* [MasaCtrl: Tuning-free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://ljzycmd.github.io/projects/MasaCtrl/)
* [Emergent Correspondence from Image Diffusion](https://diffusionfeatures.github.io/)
* [DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models](https://mc-e.github.io/project/DragonDiffusion/)
* [FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing](https://lin-chen.site/projects/freedrag/)


## Common Issues and Solutions
1) For users struggling in loading models from huggingface due to internet constraint, please 1) follow this [links](https://zhuanlan.zhihu.com/p/475260268) and download the model into the directory "local\_pretrained\_models"; 2) Run "drag\_ui.py" and select the directory to your pretrained model in "Algorithm Parameters -> Base Model Config -> Diffusion Model Path".


