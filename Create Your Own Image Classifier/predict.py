import os
import argparse
import numpy as np
import json
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model = getattr(models, checkpoint['arch'])(pretrained=False)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    img = Image.open(image).resize((256, 256))
    w, h = img.size
    left, bottom, right, top = (w - 224) / 2, (h - 224) / 2, (w + 224) / 2, (h + 224) / 2
    crop_img = img.crop((left, bottom, right, top))
    np_img = np.array(crop_img) / 255
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    norm_img = (np_img - mean) / std
    return norm_img.transpose((2, 0, 1))

def predict(image_path, model, device, topk=5):
    model.to(device).eval()
    img = process_image(image_path)
    img = torch.from_numpy(np.array([img])).float()
    with torch.no_grad():
        logps = model(img.to(device))
    probability = torch.exp(logps)
    probs, indices = probability.topk(topk, dim=1)
    idx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx}
    classes = [idx_to_class[int(i)] for i in indices[0]]
    return probs.tolist()[0], classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a pretrained neural network on an image.")
    parser.add_argument("image_path", type=str, help="Path to selected image")
    parser.add_argument("--checkpoint", type=str, help="Path to selected checkpoint", default='./checkpoint_script.pth')
    parser.add_argument("--top_K", type=int, help="Set K best prediction (default: 3)", default=3)
    parser.add_argument("--category_names", type=str, help="Path to cat_to_name JSON file", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = load_checkpoint(args.checkpoint, device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    probs, classes = predict(args.image_path, model, device, args.top_K)
    pred_labels = [cat_to_name[i] for i in classes]

    print(f'Prediction for image {args.image_path}')
    for i, label in enumerate(pred_labels):
        print(f'TOP {i+1} -> {label} with probability {probs[i]:.4f}')
