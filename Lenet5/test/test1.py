import numpy as np
from torchvision import transforms
'''
把[H,W,C]的numpy图片转换成[C,H,W]的tensor
把[0,255]的PIL.Image转换成[0,1.0]的tensor
'''
if __name__=="__main__":
    data = np.random.randint(0, 255, size=300)
    img = data.reshape(10, 10, 3)
    print(img)
    print(img.shape)
    img_tensor = transforms.ToTensor()(img)  # 转换成tensor
    print(img_tensor)
    print(img_tensor.shape)