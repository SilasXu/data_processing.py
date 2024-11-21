# **预测对象**

PCR（分类任务）：判断患者在手术时是否能达到肿瘤完全消退的状态，结果为是或否，属于分类问题。例如，将患者分为 PCR 阳性（达到完全缓解）和 PCR 阴性（未达到完全缓解）两类。
RFS（回归任务）：预测患者在癌症初次治疗结束后无癌症迹象或症状生存的时间长度，其结果是一个连续的数值，比如预测患者的无复发生存期为多少个月或多少年。

# **数据集**
临床测量特征：包括如年龄、ER、PgG、HER2、TrippleNegative Status、Chemotherapy Grade、Tumour Proliferation、Histology Type、Lymph node Status、Tumour Stage 和 Gene 等信息（文档中提及的该数据集中的临床特征）。
           这些特征是通过临床检查、检测等手段获取的，能够从不同角度反映患者的病情和身体状况。
MRI 图像特征：从磁共振图像中提取的特征，共 107 个。这些特征能够提供关于肿瘤的形态、结构、纹理等信息，从影像学角度补充对肿瘤的描述，有助于更全面地了解肿瘤的特征，从而提高预测的准确性。
            例如，肿瘤的大小、形状、信号强度等特征可能与 PCR 和 RFS 存在一定的关联。利用这些多源数据进行综合分析，可以更全面地刻画患者的病情，为准确预测 PCR 和 RFS 提供更丰富的信息基础。

# **项目结构**
project/
│
├── data/
│   ├── raw/  # 原始数据
│   ├── processed/  # 处理后的数据
│   └── external/  # 外部数据
│
├── notebooks/  # Jupyter notebooks
│
├── src/
│   ├── data_processing.py  # 数据处理模块
│   ├── feature_engineering.py  # 特征工程模块
│   ├── model_training.py  # 模型训练模块
│   ├── model_evaluation.py  # 模型评估模块
│   ├── model_prediction.py  # 模型预测模块
│   └── visualization.py  # 可视化模块
│
├── tests/  # 测试代码
│
├── requirements.txt  # 依赖包
│
└── README.md  # 项目说明