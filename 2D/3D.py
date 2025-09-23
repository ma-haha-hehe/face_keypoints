import cv2
import face_alignment
import torch
import os
import matplotlib.pyplot as plt
print(os.getcwd())
file = '2.jpg'                         # 换成你的图片路径
im = cv2.imread(file)                   # BGR
input_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转为 RGB（face-alignment 需要）

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,  # 2D 关键点（68,2）
    flip_input=False,
    device='cpu'
)
preds = fa.get_landmarks(input_img)[0]
print(preds.shape)
# (68, 2)
fig, ax = plt.subplots(figsize=(5,5))
plt.imshow(input_img)
ax.scatter(preds[:,0], preds[:,1], marker='+', c='r')
plt.show()

#3D
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
im = cv2.imread(file)
input = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
preds = fa.get_landmarks(input)[0]
import pandas as pd
df = pd.DataFrame(preds)
df.columns = ['x','y','z']
import plotly.express as px
fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z')
fig.show()
