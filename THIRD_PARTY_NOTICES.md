# Third-party notices

This repository contains code and data from other open-source projects. These parts keep their own licenses. The license of the original code in this repository is described in the [README](README.md#license).

## ColorizeNet

- Source: [rensortino/ColorizeNet](https://github.com/rensortino/ColorizeNet) by rensortino
- License: Apache License 2.0. The full text is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

Taken **unchanged** from ColorizeNet:

- `backend/src/utils/data.py`
- `backend/src/utils/ddim.py`
- `backend/src/utils/model.py`
- `backend/src/utils/diffusion.py`
- `backend/src/config.py`
- `backend/src/share.py`
- `backend/src/data/training/prompts.json` (the training list: 118,287 COCO 2017 file names, each with one of 10 short prompts)

Some of these files were themselves taken by ColorizeNet from [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) (Apache-2.0).

**Changed** by this project (adapted from ColorizeNet):

- `backend/src/models/cldm_v21.yaml`: the same model config, with `hint_channels` changed from 5 to 3.
- `backend/src/train.py`: adapted from ColorizeNet's `train.py`.
- `backend/src/data/dataset.py`: adapted from ColorizeNet's `colorization_dataset.py`.
- `backend/src/main.py`: the inference steps in `process()` follow ColorizeNet's `gradio_colorization.py` (the default added and negative prompts, the guess-mode control scales, and the LAB color transfer that keeps the input lightness). The FastAPI endpoint around it was written for this project.

## ControlNet

- Source: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang
- License: Apache License 2.0. The full text is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).
- Used for: `backend/src/cldm/` and `backend/src/ldm/`.

## Stable Diffusion

- Source: [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
- License: MIT for the code (the `ldm/` code originally comes from this repository). The Stable Diffusion 2.1 model weights use the CreativeML Open RAIL++-M license.
- Please see the upstream repository for the exact license text.
