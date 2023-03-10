# Visual ChatGPT 

**Visual ChatGPT** connects ChatGPT and a series of Visual Foundation Models to enable **sending** and **receiving** images during chatting.

See our paper: [<font size=5>Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models</font>](https://arxiv.org/abs/2303.04671)

## Demo 
<img src="./assets/demo_short.gif" width="750">

##  System Architecture 

 
<p align="center"><img src="./assets/figure.jpg" alt="Logo"></p>


## Quick Start

Use [Anaconda](https://www.anaconda.com/) and launch in Administrator Mode

```
# create a new environment & activate the new environment
conda create -n visgpt python=3.8 && conda activate visgpt

# Install PyTorch via Conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia


#  prepare the basic environments
pip install -r requirement.txt

# Clone ControlNet
git clone https://github.com/lllyasviel/ControlNet.git

# Link the files from ControlNet
mklink /D ldm ControlNet\ldm
mklink /D cldm ControlNet\cldm
mklink /D annotator ControlNet\annotator
```

download the visual foundation models and put it in `ControlNet/models`
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth
- https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth

```
# prepare your private openAI private key
set OPENAI_API_KEY={Your_Private_Openai_Key}

# create a folder to save images
mkdir image

# Start Visual ChatGPT !
python visual_chatgpt.py

# Sometimes you have to run share to be able to run it, well for me not using share doesn't work for some reason (don't share your public URL!)
python visual_chatgpt.py --share
```

The code has been set to 1 GPU only. From line `804` to `824`, it's all `device=cuda:0`

Tested on RTX3090, unfortunately got `OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB (GPU 0; 24.00 GiB total
capacity; 23.00 GiB already allocated; 0 bytes free; 23.18 GiB reserved in total by PyTorch)
If reserved memory is >> allocated memory try setting max_split_size_mb to avoid
fragmentation.` It only can be ran on a A100 LOL

This can be solved by using [https://github.com/rupeshs](rupeshs)'s suggestion.

Here, choose the tool that you want to use if you have a limited VRAM. The GPU memory usage can be found in the below table.

In the following image, Canny2Image, T2I, BLIP, and Image Caption are chosen. Around 10 VRAM is used.

![image](https://user-images.githubusercontent.com/29135514/224424232-dae66136-b846-4968-b494-c4c55d1bc60a.png)

![image](https://user-images.githubusercontent.com/29135514/224424247-5e804117-0994-4d96-b470-2b5a4a5c2484.png)


## GPU memory usage
Here we list the GPU memory usage of each visual foundation model, one can modify ``self.tools`` with fewer visual foundation models to save your GPU memory:

| Foundation Model        | Memory Usage (MB) |
|------------------------|-------------------|
| ImageEditing           | 6667              |
| ImageCaption           | 1755              |
| T2I                    | 6677              |
| canny2image            | 5540              |
| line2image             | 6679              |
| hed2image              | 6679              |
| scribble2image         | 6679              |
| pose2image             | 6681              |
| BLIPVQA                | 2709              |
| seg2image              | 5540              |
| depth2image            | 6677              |
| normal2image           | 3974              |
| Pix2Pix                | 2795              |



## Acknowledgement
We appreciate the open source of the following projects:

- HuggingFace [[Project]](https://github.com/huggingface/transformers)

- ControlNet  [[Paper]](https://arxiv.org/abs/2302.05543) [[Project]](https://github.com/lllyasviel/ControlNet)

- Stable Diffusion [[Paper]](https://arxiv.org/abs/2112.10752)  [[Project]](https://github.com/CompVis/stable-diffusion)
