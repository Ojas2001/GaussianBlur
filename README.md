# GaussianBlur

GaussianBlur is a simple Gradio web app that applies intelligent blur effects to images using AI models from Hugging Face.

## Features

* **Background Blur** – Segments the person in an image and blurs only the background (Zoom-style effect).
* Model used - ["nvidia/segformer-b0-finetuned-ade-512-512"](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
* **Depth-Based Lens Blur** – Estimates image depth and applies realistic depth-of-field blur.
* Model used - ["Intel/dpt-large"](https://huggingface.co/Intel/dpt-large)

## Tech Stack

* Python
* Gradio
* Hugging Face Transformers
* PyTorch

Link to Huggingface app space - https://huggingface.co/spaces/odeodhar/GaussianBlur
