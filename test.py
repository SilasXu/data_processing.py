import numpy as np
from sklearn.metrics import pairwise_distances


def nsknn_impute(data, k=5):
    """
    Implements No-Skip k-Nearest Neighbors (NSkNN) for missing value imputation.

    Parameters:
        data (numpy.ndarray): Input data with missing values as NaN.
        k (int): Number of nearest neighbors to use for imputation.

    Returns:
        numpy.ndarray: Data with missing values imputed.
    """
    # Create a copy of the data to avoid modifying the original
    imputed_data = data.copy()
    num_rows, num_cols = data.shape

    # Replace missing values column-wise
    for col in range(num_cols):
        # Identify rows with missing values in this column
        missing_rows = np.isnan(data[:, col])

        # Process rows with missing values
        for row in np.where(missing_rows)[0]:
            # Extract row with missing value
            target_row = data[row, :]

            # Remove rows where this column is also missing (non-skip condition)
            valid_rows = ~np.isnan(data[:, col])
            valid_data = data[valid_rows, :]

            # Calculate pairwise distances, ignoring NaNs
            distances = pairwise_distances(
                valid_data, target_row.reshape(1, -1), metric='nan_euclidean'
            ).flatten()

            # Get indices of k nearest neighbors
            neighbor_indices = np.argsort(distances)[:k]

            # Get the k nearest neighbors' values for this column
            neighbors = valid_data[neighbor_indices, col]

            # Fill missing value with the mean of k nearest neighbors
            imputed_value = np.nanmean(neighbors)
            imputed_data[row, col] = imputed_value

    return imputed_data


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 数据预处理函数，模拟生成包含缺失值的数据
def generate_missing_data(data, missing_rate):
    mask = np.random.choice([True, False], size=data.shape, p=[missing_rate, 1 - missing_rate])
    missing_data = data.copy()
    missing_data[mask] = np.nan
    return missing_data, mask

# GAIN训练函数
def train_gain(data, missing_rate, batch_size, num_epochs, generator, discriminator, optimizer_g, optimizer_d):
    # 生成包含缺失值的数据
    missing_data, mask = generate_missing_data(data, missing_rate)

    for epoch in range(num_epochs):
        for i in range(0, len(missing_data), batch_size):
            # 获取当前批次的数据
            batch_missing_data = torch.FloatTensor(missing_data[i:i + batch_size])
            batch_mask = torch.FloatTensor(mask[i:i + batch_size])

            # 训练判别器
            optimizer_d.zero_grad()

            # 用生成器生成填补后的样本
            filled_data_generated = generator(batch_missing_data)
            filled_data_generated[batch_mask] = batch_missing_data[batch_mask]

            # 组合真实未缺失数据和生成器生成的数据作为判别器输入
            real_data_batch = torch.FloatTensor(data[i:i + batch_size])
            discriminator_input = torch.cat((real_data_batch, filled_data_generated), 0)

            # 判别器输出
            discriminator_output = discriminator(discriminator_input)

            # 真实数据的标签
            real_labels = torch.ones(real_data_batch.shape[0])
            generated_labels = torch.zeros(filled_data_generated.shape[0])

            # 判别器损失
            discriminator_loss = nn.BCELoss()(discriminator_output[:real_data_batch.shape[0]], real_labels) + \
                                 nn.BCELoss()(discriminator_output[real_data_batch.shape[0]:], generated_labels)

            discriminator_loss.backward()
            optimizer_d.update()

            # 训练生成器
            optimizer_g.zero_grad()

            # 再次用生成器生成填补后的样本
            filled_data_generated = generator(batch_missing_data)
            filled_data_generated[batch_mask] = batch_missing_data[batch_mask]

            # 判别器对生成器生成数据的输出
            discriminator_output_generated = discriminator(filled_data_generated)

            # 生成器损失
            generator_loss = nn.BCELoss()(discriminator_output_generated, real_labels)

            generator_loss.backward()
            optimizer_g.update()

    return generator, discriminator

# 示例用法
if __name__ == "__main__":
    # 生成一些示例数据
    data = np.random.randn(1000, 10)

    # 设置参数
    missing_rate = 0.2
    batch_size = 32
    num_epochs = 100
    input_dim = data.shape[1]
    hidden_dim_g = 64
    hidden_dim_d = 32
    output_dim = data.shape[1]

    # 创建生成器和判别器实例
    generator = Generator(input_dim, hidden_dim_g, output_dim)
    discriminator = Discriminator(input_dim, hidden_dim_d)

    # 创建优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    # 训练GAIN
    trained_generator, trained_discriminator = train_gain(data, missing_rate, batch_size, num_epochs, generator, discriminator, optimizer_g, optimizer_d)

    # 可以使用训练好的生成器对新的包含缺失值的数据进行填补
    new_missing_data, _ = generate_missing_data(data, missing_rate)
    new_missing_data_tensor = torch.FloatTensor(new_missing_data)
    filled_data = trained_generator(new_missing_data_tensor)