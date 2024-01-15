from PIL import Image
import os
import numpy as np

def convert_tif_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的tif文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif"):
            tif_path = os.path.join(input_folder, file_name)
            
            # 生成对应的png文件名
            png_name = os.path.splitext(file_name)[0] + ".png"
            png_path = os.path.join(output_folder, png_name)

            # 打开tif文件并转换为uint8格式
            with Image.open(tif_path) as img:
                # 将图像数据类型转换为uint8
                img_array = np.array(img, dtype=np.uint8)

                # 创建新的Image对象，并设置bitdepth为8
                img_uint8 = Image.fromarray(img_array, mode=img.mode)
                img_uint8.save(png_path, "PNG")


if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder_path = "/home/yinqiang/OEM-20231226T014733Z-001/OEM/predictions"
    output_folder_path = "/home/yinqiang/OEM-20231226T014733Z-001/OEM/predictions_png"

    # 调用函数进行转换
    convert_tif_to_png(input_folder_path, output_folder_path)
