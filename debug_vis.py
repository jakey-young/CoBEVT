# -*-coding:utf-8-*-
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from einops import rearrange
import  numpy as np
tensor2 = torch.load('/home/why/YJQ/CoBEVT/opv2v/opencood/tools/临时变量/BEVemb')
# print(tensor)
tensor2 = tensor2.cpu()
# 对于RGB图像
# tensor2 = rearrange(tensor2, ' b h w c l-> (b l c) h w')
image = tensor2[0,:,:]
# for i in range(3):
#     image = torch.cat([image,tensor2[i+1,:,:]],1)
# # tensor2 = tensor2[2,:,:]
# image = np.array(tensor2)
# merged_features = np.concatenate(image, axis=0)
#
# final_tensor = image.reshape(80,64)
# print(type(tensor2))

# plt.figure()
toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
pic = toPIL(tensor2.squeeze(0))
# plt.imshow(image,cmap='gray')
# plt.colorbar()
plt.show()
# pic.save('random.jpg')

