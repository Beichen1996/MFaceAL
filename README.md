# 基于密集人脸场景下的多尺度匹配空间多样性主动标注框架

## 1. 算法描述
针对电影切片中的人脸检测和角色匹配任务的主动学习标注框架，对电影中出现的人脸密集的场景，利用人脸模型进行面部检测和特征提取，利用各个图片之间人脸的相似度和各个角色的定妆照，构建电影切片的全局角色人脸关联关系，构建密集人脸的多尺度匹配空间，选取具有高标注价值人脸的片段进行标注，显著提升标注质量和目标性能。

## 2. 环境依赖及安装
该框架所需的环境依赖如下：

- torch == 1.6.0
- torchvision == 0.9.1
- numpy == 1.19.2
- scikit-learn == 0.23.2
- opencv-python == 4.6.0

建议使用anaconda或pip配置环境。例如：
```
pip install torch==1.6.0
pip install torchvision == 0.9.1
pip install numpy==1.19.2
pip install scikit-learn==0.23.2
pip install opencv-python==4.6.0
```

## 3. 运行示例

### 代码准备

1. 针对ubuntu系统，设置库的路径
```
ubuntu:
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{%存放该文件夹的绝对地址}/activelearning_face/utils/lib/ubuntu
centos:
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{%存放该文件夹的绝对地址}/activelearning_face/utils/lib/centos
```

2. 下载部分预训练模型到 “./seetaface/model” 下
```
    https://pan.baidu.com/share/init?surl=LlXe2-YsUxQMe-MLzhQ2Aw 提取码：ngne
```

### 模型训练使用
1. 运行以下命令，可以对“test_tt1699513_jd_1_347_356”文件下的示例样本进行主动学习采样，采样结果保存在“selected”文件夹下
```
python main.py ./test_tt1699513_jd_1_347_356/
```


## 4. 论文/专利成果
> Beichen Zhang, Liang Li, Zheng-Jun Zha and Qingming Huang. Contrastive Cross-modal Representation Learning Based Active Learning for Visual Question Answer. Chinese Journal of Computers
