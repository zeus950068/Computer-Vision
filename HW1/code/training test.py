import numpy as np
import cv2

# 生成一个3x3的二维卷积核
kernel = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]])

# 创建一个3x3x3的三维输入数据（假设有3个深度层）
# input_data = np.array([[[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]],
#                        [[9, 8, 7],
#                         [6, 5, 4],
#                         [3, 2, 1]],
#                        [[2, 3, 4],
#                         [5, 6, 7],
#                         [8, 9, 10]]])
input_data =cv2.imread(r"c:\Users\zeus9\OneDrive - NTHU\Courses\2023_fall\computer_vision\HW1\data\einstein.bmp")

print(input_data.shape)
print(input_data)

# 获取卷积核和输入数据的维度信息
kernel_height, kernel_width = kernel.shape
depth, input_height, input_width = input_data.shape

# 计算卷积后的输出尺寸
output_height = input_height - kernel_height + 1
output_width = input_width - kernel_width + 1

# 初始化输出数据
output_data = np.zeros((depth, output_height, output_width))

# 对每个深度层执行卷积操作
for d in range(depth):
    for i in range(output_height):
        for j in range(output_width):
            # 获取输入数据中与卷积核对应的切片
            input_slice = input_data[d, i:i+kernel_height, j:j+kernel_width]
            # print("input_slice\n", input_slice)
            
            # 使用np.pad填充切片
            padded_slice = np.pad(input_slice, ((0, 0), (0, 0)), mode='constant')
            
            # 计算卷积操作并将结果存储在输出数据中
            output_data[d, i, j] = np.sum(padded_slice * kernel)

# 输出卷积后的结果
print("卷积后的输出数据：")
print(output_data)
