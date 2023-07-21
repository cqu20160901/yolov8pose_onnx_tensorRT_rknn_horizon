# yolov8pose_onnx_tensorRT_rknn_horizon
yolov8pose 部署版本，便于移植不同平台（onnx、tensorRT、rknn、Horizon）。

***特别说明：本示例提供的代码只适用按照本链接提供的方式导出的onnx，导出onnx方式参考后文链接。本示例中模型的训练使用的数据不多，模型效果无法保证，只是用来测试部署用的，如果换其他图像可能存在检测不到的现象。***

# 文件夹结构说明

yolov8pose_onnx：onnx模型、测试图像、测试结果、pytorch测试结果、测试demo脚本

yolov8pose_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolov8pose_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

yolov8pose_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 测试结果

pytyorch 测试结果

![image](https://github.com/cqu20160901/yolov8pose_onnx_tensorRT_rknn_horizon/blob/main/yolov8pose_onnx/test_pytorch_result.jpg)

onnx测试结果

![image](https://github.com/cqu20160901/yolov8pose_onnx_tensorRT_rknn_horizon/blob/main/yolov8pose_onnx/test_onnx_result.jpg)

（注：图片来源coco数据集）

说明：推理测试预处理没有考虑等比率缩放，激活函数 SiLU 用 Relu 进行了替换。由于模型训练使用的数据并不多，且迭代的次数不多，效果并不是很好，仅供测试流程用。



# 相关链接
yolov8 检测部署

## yolov8 检测部署

部署导出方高效方式参考 [类似本实例中导出检测方式](https://blog.csdn.net/zhangqian_1/article/details/128918268)

官方导出onnx方式板端部署方式参考 [官方导出onnx方式部署](https://blog.csdn.net/zhangqian_1/article/details/130754564) 

rknn的板端C++部署参考 [C++部署](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon)

## yolov8seg 部署

yolov8seg 部署 [https://github.com/cqu20160901/yolov8seg_onnx_tensorRT_rknn_horizon]



