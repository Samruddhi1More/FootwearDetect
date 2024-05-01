import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
#explicitly mentioning classes
label2id =  {
    'boots': 0,
    'flip_flops': 1,
    'loafers': 2,
    'sandals': 3,
    'sneakers': 4,
    'soccer_shoes': 5
}

id2label = {id:label for label,id in label2id.items()}

#Load a pretrained model for image classification
#Load  fine-tuned weights
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name,
                                                  num_labels=6,
                                                  id2label=id2label,
                                                  label2id=label2id)
# Debugging: Print the loaded model configuration
print(model.config)

# Load fine-tuned weights (check if the file path is correct)
try:
    model.load_state_dict(torch.load("./Model Weights/footweartype_VIT.pth", map_location=torch.device("cpu")))
    print("Fine-tuned weights loaded successfully!")
except FileNotFoundError:
    print("Error: Fine-tuned weights file not found!")

class DetectClasses:
    def __init__(self):
        self.data_transformations = transforms.Compose([
            transforms.Resize(size=(224,224)),  #re-sizing
            transforms.ToTensor()])
        
    def predict(self, image_path):
        image = Image.open(image_path)
        test_input = self.data_transformations(image)

        with torch.no_grad():
            test_img = test_input.unsqueeze(dim=0)
            logits = model(test_img).logits
        predicted_label = logits.argmax(-1).item()

        pred = model.config.id2label[predicted_label]

        return pred