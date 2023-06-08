# image-inpainting-with-context-flow-network
image inpainting with context flow network pytorch实现

# 使用说明
MASK来自于https://nv-adlr.github.io/publication/partialconv-inpainting

修改configs/config.yaml文件指定目录和相关参数即可

先用onlycoarse.py训练粗模型，然后使用main.py进行整体训练。粗模型训练可适当调大batch_size。

部署暂未开发。


