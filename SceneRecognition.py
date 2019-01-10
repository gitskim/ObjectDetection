# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import numpy as np
from pprint import pprint
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img_name = '12.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# load test images
folder = '/dvmm-filer2/projects/Hearst/keyframes2_dec_20'

predictions_to_return = 5
correct = 1
total = 1
pic_type = ".png"
counter = 0
dictionary = {}
mflag = False

for root, dirs, files in os.walk(folder):
    if mflag:
        break

    for mfile in [f for f in files if f.endswith(pic_type)]:
        filename = os.path.join(root, mfile)

        x = [None]*5
        counter += 1
        print(counter)

        if counter == 5:
            mflag = True
            break

        img = Image.open(os.path.join(root, mfile))
        input_img = V(centre_crop(img).unsqueeze(0))
        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        total += 1
        # output the prediction
        for i in range(0, predictions_to_return):
            x[i] = '{:.3f}:{}'.format(probs[i], classes[idx[i]])

            # save the results
            namelist = filename.split('/')
            namelist2 = [""] * 3

            for item in namelist:
                if not item:
                    continue

                if item.endswith(".MXF") or item.endswith(".MP4") or item.endswith(".mp4"):
                    namelist2[1] = item

                elif item.endswith(".png"):
                    namelist2[2] = item

                else:
                    namelist2[0] = namelist2[0] + '/' + item

            first = namelist2[0]
            second = namelist2[1]
            third = namelist2[2]
            fourth = x
            # d = {namelist2[0]: {namelist2[1]: {namelist2[2]: val}}}

            if first in dictionary:
                if second in dictionary[first]:
                    if third in dictionary[first][second]:
                        dictionary[first][second][third].append(fourth)
                    else:
                        dictionary[first][second][third] = fourth
                else:
                    dictionary[first][second] = {third: fourth}
            else:
                dictionary[first] = {second: {third: fourth}}

            pprint(dictionary)


result_file_name = 'scene_recognition_result_jan_10.data'
with open(result_file_name, 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

