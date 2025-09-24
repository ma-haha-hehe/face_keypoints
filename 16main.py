import torch.nn as nn
import torch
from torch import inference_mode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import numpy as np,os,glob
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
from copy import deepcopy
device ='cuda' if torch.cuda.is_available() else 'cpu'

root_dir = 'P1_Facial_Keypoints/data/training/'
all_img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
data = pd.read_csv('P1_Facial_Keypoints/data/training_frames_keypoints.csv')

class Facesdata(Dataset):
    def __init__(self,df):
        super(Facesdata).__init__()
        self.df = df
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    def __len__(self):
        return len(self.df)
    def __getitem__(self,ix):
        img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)/255
        kp = deepcopy(self.df.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()  # 除以宽W
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()  # 除以高H
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2
    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img.to(device)

    #定义函数加载图像
    def load_img(self, ix):
        img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (224, 224))
        return img

#拆分测试和训练数据集
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size=0.2,random_state=101)

train_dataset = Facesdata(train.reset_index(drop=True))
test_dataset = Facesdata(test.reset_index(drop=True))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#model = models.vgg16(pretrained=True)
#print(model)

def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()
    )
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    return model.to(device), criterion, optimizer

model, criterion, optimizer = get_model()
print(model)

def train_batch(img,kps,model,optimizer,criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss
@torch.no_grad()
def valid_batch(img,kps,model,criterion):
    model.eval()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps,loss
#start train
train_loss,test_loss = [],[]
n_epoch = 10
for epoch in range(n_epoch):
    print(f"epoch {epoch+1}/10")
    epoch_train_loss,epoch_test_loss=0,0#初始化本epoch的训练损失和测试损失
    for ix, (img, kps) in enumerate(train_loader):
        loss = train_batch(img,kps,model,optimizer,criterion)
        epoch_train_loss+=loss.item()
    epoch_train_loss /= (ix+1)
    for ix, (img, kps) in enumerate(test_loader):
        ps,loss = valid_batch(img,kps,model,criterion)
        epoch_test_loss+=loss.item()
    epoch_test_loss /= (ix+1)

    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)

epochs = np.range(10) + 1
plt.plot(epochs,train_loss,'bo',label = 'Training loss')
plt.plot(epochs,test_loss,'ro',label = 'Tess loss')
plt.titel('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()


ix = 20
plt.figure(figsize=(10,10))

plt.subplot(121)
plt.title('Original image')
im = test_dataset.load_img(ix)
plt.imshow(im)
plt.grid(False)

plt.subplot(122)
plt.title('Image with facial keypoints')
x, _ = test_dataset[ix]
plt.imshow(im)
kp = model(x[None]).flatten().detach().cpu()
plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
plt.grid(False)

plt.show()

