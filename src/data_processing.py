import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Load data from an Excel file and handle missing values.

    Parameters:
    file_path (str): Path to the data file.

    Returns:
    data (DataFrame): Processed data.
    """
    data = pd.read_excel(file_path).drop(columns=['ID'])
    data = np.where(data == 999, np.nan, data)
    data = pd.DataFrame(data)
    return data


def clean_data_except_gene(data):
    """
    Clean data by dropping rows with missing values except for the gene column.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Cleaned data.
    """
    columns_except_gene = data.columns.difference([12])
    data = data.dropna(subset=columns_except_gene)
    return data


def fill_missing_values_with_lr(data_gene_missing, data_gene_not_missing):
    """
    Predict and fill missing values in the gene column using a model.

    Parameters:
    data_gene_missing (DataFrame): Data with missing gene values.
    data_gene_not_missing (DataFrame): Data without missing gene values.

    Returns:
    data_filled (DataFrame): Data with filled gene values.
    """
    # Prepare input variables and target variable
    X = data_gene_not_missing.drop(columns=[12])
    y = data_gene_not_missing[12]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    # Train logistic regression model
    log_reg = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
    log_reg.fit(X_lda, y)

    # Output model accuracy
    accuracy = cross_val_score(log_reg, X_lda, y, cv=5, scoring='accuracy').mean()
    print(f"Model accuracy: {accuracy:.4f}")

    # Process data with missing values
    X_missing = data_gene_missing.drop(columns=[12])
    X_missing_scaled = scaler.transform(X_missing)
    X_missing_lda = lda.transform(X_missing_scaled)

    # Predict missing values
    y_missing_pred = log_reg.predict(X_missing_lda)

    # Fill missing values
    data_gene_missing.loc[:, 12] = y_missing_pred
    data_filled = pd.concat([data_gene_not_missing, data_gene_missing])

    return data_filled


def split_features_and_targets(data, classification_target_column, regression_target_column):
    """
    Split data into features and two targets for machine learning.

    Parameters:
    data (DataFrame): Input data.
    classification_target_column (str or int): Column name or index for classification target.
    regression_target_column (str or int): Column name or index for regression target.

    Returns:
    X (DataFrame): Feature data.
    y_classification (Series): Classification target data.
    y_regression (Series): Regression target data.
    """
    X = data.drop(columns=[classification_target_column, regression_target_column])
    y_classification = data[classification_target_column]
    y_regression = data[regression_target_column]
    return X, y_classification, y_regression


def split_data_by_gene(data, gene_column):
    """
    Split data into two parts based on whether the gene column is missing.

    Parameters:
    data (DataFrame): Input data.
    gene_column (str): Name of the gene column.

    Returns:
    data_gene_missing (DataFrame): Data with missing gene values.
    data_gene_not_missing (DataFrame): Data without missing gene values.
    """
    data_gene_missing = data[data[gene_column].isna()]
    data_gene_not_missing = data[~data[gene_column].isna()]
    return data_gene_missing, data_gene_not_missing


class NeuralNet(nn.Module):
    """
    Defines a simple neural network model for filling missing gene data.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        l2_lambda (float): L2 regularization coefficient.
    """

    def __init__(self, input_size, l2_lambda=0.0):
        """
        Initializes the neural network model.

        Parameters:
            input_size (int): Dimension of input features.
            l2_lambda (float): L2 regularization coefficient.
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l2_lambda = l2_lambda

    def forward(self, x):
        """
        Forward propagation function.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def l2_regularization(self):
        """
        Computes the L2 regularization term.

        Returns:
            torch.Tensor: L2 regularization term.
        """
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.l2_lambda * l2_reg


def fill_missing_values_with_nn(data_gene_missing, data_gene_not_missing, param_grid, num_epochs=50, n_splits=5):
    """
    Fills missing gene data using a neural network with grid search and cross-validation.

    Parameters:
        data_gene_missing (pd.DataFrame): Data with missing gene values.
        data_gene_not_missing (pd.DataFrame): Data without missing gene values.
        param_grid (dict): Parameter grid for grid search.
        num_epochs (int): Number of training epochs.
        n_splits (int): Number of cross-validation splits.

    Returns:
        pd.DataFrame: Data with filled gene values.
    """
    # Separate features and target variable
    X = data_gene_not_missing.drop(columns=[12])
    y = data_gene_not_missing[12]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # Grid search and cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = 0
    best_params = {}

    for l2_lambda in param_grid['l2_lambda']:
        for lr in param_grid['lr']:
            fold_accuracies = []
            for train_index, val_index in kf.split(X_tensor):
                X_train, X_val = X_tensor[train_index], X_tensor[val_index]
                y_train, y_val = y_tensor[train_index], y_tensor[val_index]

                # Create data loader
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=True)

                # Define neural network model
                model = NeuralNet(input_size=X.shape[1], l2_lambda=l2_lambda)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Train model
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) + model.l2_regularization()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Validate model
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val).numpy().flatten()
                    y_val_pred = (y_val_pred > 0.5).astype(int)
                    y_val_true = y_val.numpy().flatten()
                    accuracy = accuracy_score(y_val_true, y_val_pred)
                    fold_accuracies.append(accuracy)

            avg_accuracy = np.mean(fold_accuracies)
            if avg_accuracy > best_score:
                best_score = avg_accuracy
                best_params = {'l2_lambda': l2_lambda, 'lr': lr}

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score:.4f}")

    # Retrain model with best parameters
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=10, shuffle=True)
    model = NeuralNet(input_size=X.shape[1], l2_lambda=best_params['l2_lambda'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels) + model.l2_regularization()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Fill missing data
    model.eval()
    with torch.no_grad():
        X_missing = data_gene_missing.drop(columns=[12])
        X_missing_scaled = scaler.transform(X_missing)
        X_missing_tensor = torch.tensor(X_missing_scaled, dtype=torch.float32)
        y_missing_pred = model(X_missing_tensor)
        data_gene_missing.loc[:, 12] = y_missing_pred.numpy().flatten()

    data_filled = pd.concat([data_gene_not_missing, data_gene_missing])
    return data_filled


def fill_missing_values_with_voting(data_gene_missing, data_gene_not_missing, n_splits=5):
    """
    Fill missing gene data using a voting method and perform cross-validation.

    Parameters:
        data_gene_missing (pd.DataFrame): DataFrame with missing gene data.
        data_gene_not_missing (pd.DataFrame): DataFrame without missing gene data.
        n_splits (int): Number of cross-validation folds.

    Returns:
        pd.DataFrame: DataFrame with filled gene data.
    """
    # Prepare input variables and target variable
    X = data_gene_not_missing.drop(columns=[12])
    y = data_gene_not_missing[12]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    # Define voting classifier
    clf1 = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = SVC(probability=True)
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')

    # Train voting classifier
    voting_clf.fit(X_lda, y)

    # Output model accuracy
    accuracy = cross_val_score(voting_clf, X_lda, y, cv=n_splits, scoring='accuracy').mean()
    print(f"Model accuracy: {accuracy:.4f}")

    # Process data with missing values
    X_missing = data_gene_missing.drop(columns=[12])
    X_missing_scaled = scaler.transform(X_missing)
    X_missing_lda = lda.transform(X_missing_scaled)

    # Predict missing values
    y_missing_pred = voting_clf.predict(X_missing_lda)

    # Fill missing values
    data_gene_missing.loc[:, 12] = y_missing_pred
    data_filled = pd.concat([data_gene_not_missing, data_gene_missing])

    return data_filled


def fill_missing_values(lr=False, nn=False, voting=False, gain=False):
    if lr:
        data_processed = fill_missing_values_with_lr(data_gene_missing, data_gene_not_missing)
    elif nn:
        param_grid = {'l2_lambda': [0.0, 0.01, 0.1], 'lr': [0.01, 0.001]}
        data_processed = fill_missing_values_with_nn(data_gene_missing, data_gene_not_missing, param_grid)
    elif voting:
        data_processed = fill_missing_values_with_voting(data_gene_missing, data_gene_not_missing)
    elif gain:
        param_grid = {'lr': [0.01, 0.001], 'batch_size': [10, 20]}
        data_processed = fill_missing_values_with_gain(data_gene_missing, data_gene_not_missing, param_grid)
    return data_processed

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

def l2_regularization(model, lambda_l2):
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    return lambda_l2 * l2_reg

def train_gain(data, batch_size, num_epochs, generator, discriminator, optimizer_g, optimizer_d, lambda_l2=0.01):
    mask = ~np.isnan(data)

    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            batch_data = torch.FloatTensor(data[i:i + batch_size])
            batch_mask = torch.BoolTensor(mask[i:i + batch_size])

            optimizer_d.zero_grad()
            filled_data_generated = generator(batch_data)
            filled_data_generated[batch_mask] = batch_data[batch_mask]
            real_data_batch = torch.FloatTensor(data[i:i + batch_size])
            discriminator_input = torch.cat((real_data_batch, filled_data_generated), 0)
            discriminator_output = discriminator(discriminator_input)
            real_labels = torch.ones(real_data_batch.shape[0], 1)
            generated_labels = torch.zeros(filled_data_generated.shape[0], 1)
            discriminator_loss = nn.BCELoss()(discriminator_output[:real_data_batch.shape[0]], real_labels) + \
                                 nn.BCELoss()(discriminator_output[real_data_batch.shape[0]:], generated_labels)
            discriminator_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            filled_data_generated = generator(batch_data)
            filled_data_generated[batch_mask] = batch_data[batch_mask]
            discriminator_output_generated = discriminator(filled_data_generated)
            generator_loss = nn.BCELoss()(discriminator_output_generated, real_labels) + l2_regularization(generator, lambda_l2)
            generator_loss.backward()
            optimizer_g.step()

    return generator, discriminator
def fill_missing_values_with_gain(data_gene_missing, data_gene_not_missing, param_grid, num_epochs=100, n_splits=5):
    data = pd.concat([data_gene_missing, data_gene_not_missing])
    data_values = data.values

    input_dim = data_values.shape[1]
    hidden_dim_g = 64
    hidden_dim_d = 32
    output_dim = data_values.shape[1]

    best_score = 0
    best_params = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            fold_accuracies = []
            for train_index, val_index in kf.split(data_values):
                train_data, val_data = data_values[train_index], data_values[val_index]

                generator = Generator(input_dim, hidden_dim_g, output_dim)
                discriminator = Discriminator(input_dim, hidden_dim_d)
                optimizer_g = optim.Adam(generator.parameters(), lr=lr)
                optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

                trained_generator, _ = train_gain(train_data, batch_size, num_epochs, generator, discriminator, optimizer_g, optimizer_d)

                val_data_tensor = torch.FloatTensor(val_data)
                mask = ~np.isnan(val_data)
                val_data_filled = trained_generator(val_data_tensor).detach().numpy()
                val_data_filled[mask] = val_data[mask]

                accuracy = accuracy_score(~np.isnan(val_data), ~np.isnan(val_data_filled))
                fold_accuracies.append(accuracy)

            avg_accuracy = np.mean(fold_accuracies)
            if avg_accuracy > best_score:
                best_score = avg_accuracy
                best_params = {'lr': lr, 'batch_size': batch_size}

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score:.4f}")

    generator = Generator(input_dim, hidden_dim_g, output_dim)
    discriminator = Discriminator(input_dim, hidden_dim_d)
    optimizer_g = optim.Adam(generator.parameters(), lr=best_params['lr'])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=best_params['lr'])

    trained_generator, _ = train_gain(data_values, best_params['batch_size'], num_epochs, generator, discriminator, optimizer_g, optimizer_d)

    new_missing_data_tensor = torch.FloatTensor(data_gene_missing.values)
    filled_data = trained_generator(new_missing_data_tensor).detach().numpy()

    data_gene_missing_filled = data_gene_missing.copy()
    data_gene_missing_filled.loc[:, :] = filled_data

    data_filled = pd.concat([data_gene_not_missing, data_gene_missing_filled])
    return data_filled


# Load data
file_path = '../data/raw/TrainDataset2024.xls'
data_loaded = load_data(file_path)

# Clean data
data_cleaned_except_gene = clean_data_except_gene(data_loaded)

# Split features and target variables
X, y_classification, y_regression = split_features_and_targets(data_cleaned_except_gene, 0, 1)

# Split data based on whether the gene column is missing
data_gene_missing, data_gene_not_missing = split_data_by_gene(X, 12)

# Fill missing gene values
data_processed = fill_missing_values(lr=False, nn=False, voting=False, gain=True)

# Add classification and regression targets back to the processed data
data_processed['classification_target'] = y_classification.values
data_processed['regression_target'] = y_regression.values

# Save the processed data to a CSV file
output_file_path = '../data/processed/ProcessedDataset2024.csv'
data_processed.to_csv(output_file_path, index=False)
