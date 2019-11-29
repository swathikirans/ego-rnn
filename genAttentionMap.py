import numpy as np
from torchvision import transforms
import cv2
from objectAttentionModelConvLSTM import *
from attentionMapModel import attentionMap
from PIL import Image

####################Model definition###############################
num_classes = 61 # Classes in the pre-trained model
mem_size = 512
model_state_dict = 'models/best_model_state_dict_rgb_split2.pth' # Weights of the pre-trained model

model = attentionModel(num_classes=num_classes, mem_size=mem_size)
model.load_state_dict(torch.load(model_state_dict))
model_backbone = model.resNet
attentionMapModel = attentionMap(model_backbone).cuda()
attentionMapModel.train(False)
for params in attentionMapModel.parameters():
    params.requires_grad = False
###################################################################

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess1 = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
])

preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    normalize])


fl_name_in = 'test_image.jpg'
fl_name_out = 'test_image_attention.jpg'
img_pil = Image.open(fl_name_in)
img_pil1 = preprocess1(img_pil)
img_size = img_pil1.size
size_upsample = (img_size[0], img_size[1])
img_tensor = preprocess2(img_pil1)
img_variable = Variable(img_tensor.unsqueeze(0).cuda())
img = np.asarray(img_pil1)
attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
cv2.imwrite(fl_name_out, attentionMap_image)