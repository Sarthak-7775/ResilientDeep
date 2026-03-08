dATASET dOWNLAODED from----
https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset?resource=download







The models/ directory is strictly for storing weights (the mathematical parameters your neural network learns), not the Python code itself.

Think of src/modules/model.py as the "skeleton" of your network, and the files in the models/ folder as the "brain" that gets inserted into that skeleton.

Here is exactly what goes into each subfolder and how to interact with them.

1. models/pre_trained/
This folder holds weights created by other researchers before you do any fine-tuning.

What to put here:


Baseline Weights: As per your project roadmap, you will adapt pre-trained weights from Chhabra et al. and Gao et al.. If you obtain their .pth files, they go here.

ESRGAN Weights: If you use the basicsr library for AI upscaling, it often requires downloading a pre-trained generator file (e.g., RealESRGAN_x4plus.pth). Save that here so your upscale.py script can access it offline.

Backbone Weights: PyTorch usually downloads ResNet18 weights automatically, but if you are working on an offline hardware node, you would manually place resnet18-f37072fd.pth here.

2. models/checkpoints/
This is where your program saves your specific model as it learns from your newly created attack dataset.

What to put here:

latest_model.pth: Saved at the end of every epoch so you can resume if your hardware crashes.

best_model.pth: Saved only when the model achieves a new highest F1-score or lowest loss.

