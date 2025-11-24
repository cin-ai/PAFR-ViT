import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torch import nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform(tensor, height=37, width=37):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, -1)
    result = result.permute(0, 3, 1, 2)
    return result


def load_model(config_name, img_size, pretrained_dir, model_path, device):
    config = CONFIGS[config_name]
    config.split = 'overlap'
    config.img_size = img_size
    config.pretrained_dir = pretrained_dir

    model = VisionTransformer(config, img_size=img_size, zero_head=True, num_classes=200)
    pretrained_model = torch.load(model_path, map_location=device)['model']
    model.load_state_dict(pretrained_model)
    model.to(device)
    model.eval()
    return model


class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transformer = model.transformer
        self.part_head = model.part_head

    def forward(self, x):
        out = self.transformer(x)
        logits = self.part_head(out[:, 0])
        return logits


def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_resized = transforms.Resize((600, 600))(img)
    img_cropped = transforms.CenterCrop((img_size, img_size))(img_resized)
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor, img_cropped


def predict(image_path, model, device, img_size):
    input_tensor, _ = preprocess_image(image_path, img_size)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


def show_gradcam(image_path, model, target_layer, device, img_size, class_idx):
    model.eval()
    input_tensor, processed_img = preprocess_image(image_path, img_size)
    input_tensor = input_tensor.to(device)

    targets = [ClassifierOutputTarget(class_idx)]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        use_cuda=(device.type == 'cuda'),
        reshape_transform=lambda x: reshape_transform(x, height=37, width=37)
    )

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    grayscale_cam = 1 - grayscale_cam

    rgb_img = np.array(processed_img).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite("gradcam_result.jpg", visualization)


if __name__ == '__main__':
    from models.modeling_PAFR import VisionTransformer, CONFIGS

    model_path = '/home/YCX/Transfg/output/PAFR-ViT_checkpoint.bin'

    # from models.modeling_transfg import VisionTransformer, CONFIGS
    # model_path = '/home/YCX/Transfg/output/TransFG_checkpoint.bin'

    image_path = '/home/YCX/datasets/CUB_200_2011/images/145.Elegant_Tern/Elegant_Tern_0004_150948.jpg'
    config_name = 'ViT-B_16'
    pretrained_dir = '/home/YCX/datasets/ViT-B_16.npz'

    img_size = 448
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_raw = load_model(config_name, img_size, pretrained_dir, model_path, device)
    model = ViTWrapper(model_raw).to(device)
    # print(model)

    pred_class = predict(image_path, model, device, img_size)
    print(f"Predicted class: {pred_class}")

    target_layer = model_raw.transformer.encoder.layer[-1].attention_norm
    show_gradcam(image_path, model, target_layer, device, img_size, class_idx=pred_class)