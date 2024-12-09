import os
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

# 预处理数据集，将图像中心裁剪为64x64大小，保存到imgs文件夹中。节约后续训练时间
## dataloader从文件夹中读取图片时，由于数据集过大，预处理需要大量时间，会导致CPU瓶颈问题，浪费GPU资源。所以提前完成预处理工作，节约训练时间
transform = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor()
])

# 输入和输出文件夹路径
input_folder = './img_align_celeba'  # 输入图片文件夹
output_folder = './imgs'  # 输出图片文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 遍历图片
for filename in os.listdir(input_folder):
    # 图片路径
    img_path = os.path.join(input_folder, filename)
    
    # 判断是否是图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        img = Image.open(img_path)
        img_tensor = transform(img)
        img_transformed = F.to_pil_image(img_tensor)        # 将Tensor转回为图片

        # 保存转换后的图片到新的文件夹
        output_path = os.path.join(output_folder, filename)
        img_transformed.save(output_path)

        #print(f"处理并保存图片: {output_path}")            # 由于数据集较大的原因，所以不建议在终端显示打印。可节约IO资源，节省大量时间

print("所有图片处理完成！")
