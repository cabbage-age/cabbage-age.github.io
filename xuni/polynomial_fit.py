"""
多项式拟合演示实验 - 核心功能模块

这个模块提供了多项式拟合演示实验的核心功能，包括数据生成、特征转换、模型训练和误差计算等功能。
主要用于展示机器学习中的欠拟合和过拟合现象，以及训练数据量对模型性能的影响。

主要功能：
1. 生成模拟数据集
2. 多项式特征转换
3. 模型训练和预测
4. 误差计算和评估
5. 学习曲线生成
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

def generate_data(n_samples=20, noise=0.1):
    """
    生成模拟数据集
    
    参数:
        n_samples (int): 样本数量，默认为20
        noise (float): 噪声强度，默认为0.1
        
    返回:
        tuple: (X, y)
            - X: 特征矩阵，形状为(n_samples, 1)
            - y: 目标值，形状为(n_samples,)
            
    示例:
        >>> X, y = generate_data(n_samples=50, noise=0.2)
        >>> print(X.shape, y.shape)
        (50, 1) (50,)
    """
    np.random.seed(42)  # 设置随机种子以确保结果可复现
    X = np.linspace(-3, 3, n_samples)  # 在[-3, 3]区间生成均匀分布的点
    # 使用更复杂的函数来生成数据，使其更容易展示欠拟合和过拟合
    y = 0.5 * np.sin(2 * X) + 0.3 * np.cos(3 * X) + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y

def create_polynomial_features(X, degree):
    """
    创建多项式特征
    
    参数:
        X (ndarray): 输入特征矩阵，形状为(n_samples, 1)
        degree (int): 多项式阶数
        
    返回:
        ndarray: 转换后的特征矩阵，包含多项式特征
        
    示例:
        >>> X = np.array([[1], [2], [3]])
        >>> X_poly = create_polynomial_features(X, degree=2)
        >>> print(X_poly.shape)
        (3, 3)
    """
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

def train_model(X_poly, y, train_idx):
    """
    训练模型
    
    参数:
        X_poly (ndarray): 多项式特征矩阵
        y (ndarray): 目标值
        train_idx (ndarray): 训练集索引
        
    返回:
        LinearRegression: 训练好的模型
    """
    # 使用Ridge回归，alpha参数控制正则化强度
    # 当degree较高时，减小alpha以获得更好的拟合效果
    model = Ridge(alpha=0.0001)
    model.fit(X_poly[train_idx], y[train_idx])
    return model

def calculate_errors(model, X_poly, y, train_idx, test_idx):
    """
    计算训练集和测试集的均方误差
    
    参数:
        model (LinearRegression): 训练好的模型
        X_poly (ndarray): 多项式特征矩阵
        y (ndarray): 目标值
        train_idx (ndarray): 训练集索引
        test_idx (ndarray): 测试集索引
        
    返回:
        tuple: (train_error, test_error)
            - train_error: 训练集均方误差
            - test_error: 测试集均方误差
            
    示例:
        >>> train_error, test_error = calculate_errors(model, X_poly, y, train_idx, test_idx)
        >>> print(f"训练误差: {train_error:.4f}, 测试误差: {test_error:.4f}")
    """
    train_error = mean_squared_error(y[train_idx], model.predict(X_poly[train_idx]))
    test_error = mean_squared_error(y[test_idx], model.predict(X_poly[test_idx]))
    return train_error, test_error

def generate_plot_data(model, X, y, train_idx, test_idx, degree):
    """
    生成绘图所需的数据
    
    参数:
        model (LinearRegression): 训练好的模型
        X (ndarray): 原始特征矩阵
        y (ndarray): 目标值
        train_idx (ndarray): 训练集索引
        test_idx (ndarray): 测试集索引
        degree (int): 多项式阶数
        
    返回:
        tuple: (train_data, test_data, curve_data, train_error, test_error)
            - train_data: 训练数据点
            - test_data: 测试数据点
            - curve_data: 拟合曲线数据
            - train_error: 训练误差
            - test_error: 测试误差
    """
    # 生成预测曲线
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_plot_poly = create_polynomial_features(X_plot, degree)
    y_plot = model.predict(X_plot_poly)
    
    # 计算误差
    train_error, test_error = calculate_errors(model, X_poly, y, train_idx, test_idx)
    
    # 准备绘图数据
    train_data = {
        'x': X[train_idx].flatten().tolist(),
        'y': y[train_idx].tolist(),
        'name': '训练数据',
        'mode': 'markers',
        'marker': {'color': 'blue'}
    }
    
    test_data = {
        'x': X[test_idx].flatten().tolist(),
        'y': y[test_idx].tolist(),
        'name': '测试数据',
        'mode': 'markers',
        'marker': {'color': 'red'}
    }
    
    curve_data = {
        'x': X_plot.flatten().tolist(),
        'y': y_plot.tolist(),
        'name': '拟合曲线',
        'mode': 'lines',
        'line': {'color': 'green'}
    }
    
    return train_data, test_data, curve_data, train_error, test_error

def generate_learning_curve_data(n_samples_list, degree, noise=0.1, test_size=0.2):
    """
    生成学习曲线数据
    
    参数:
        n_samples_list (list): 不同训练集大小的列表
        degree (int): 多项式阶数
        noise (float): 噪声强度
        test_size (float): 测试集比例
        
    返回:
        dict: 包含学习曲线数据的字典
    """
    train_errors = []
    test_errors = []
    
    for n_samples in n_samples_list:
        # 生成数据
        X, y = generate_data(n_samples=n_samples, noise=noise)
        
        # 划分训练集和测试集
        n_test = int(n_samples * test_size)
        test_idx = np.random.choice(n_samples, n_test, replace=False)
        train_idx = np.array([i for i in range(n_samples) if i not in test_idx])
        
        # 创建多项式特征
        X_poly = create_polynomial_features(X, degree)
        
        # 训练模型
        model = train_model(X_poly, y, train_idx)
        
        # 计算误差
        train_error, test_error = calculate_errors(model, X_poly, y, train_idx, test_idx)
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return {
        'n_samples': n_samples_list,
        'train_errors': train_errors,
        'test_errors': test_errors
    }

def generate_comparison_plots(n_samples_list, degree, noise=0.1):
    """
    生成不同数据量下的拟合效果对比图数据
    
    参数:
        n_samples_list (list): 不同训练集大小的列表
        degree (int): 多项式阶数
        noise (float): 噪声强度
        
    返回:
        list: 包含每个数据量下的拟合曲线数据的列表
    """
    plot_data = []
    
    for n_samples in n_samples_list:
        # 生成数据
        X, y = generate_data(n_samples=n_samples, noise=noise)
        
        # 划分训练集和测试集
        n_test = int(n_samples * 0.2)
        test_idx = np.random.choice(n_samples, n_test, replace=False)
        train_idx = np.array([i for i in range(n_samples) if i not in test_idx])
        
        # 创建多项式特征
        X_poly = create_polynomial_features(X, degree)
        
        # 训练模型
        model = train_model(X_poly, y, train_idx)
        
        # 生成绘图数据
        train_data, test_data, curve_data, train_error, test_error = generate_plot_data(
            model, X, y, train_idx, test_idx, degree
        )
        
        plot_data.append({
            'n_samples': n_samples,
            'train_data': train_data,
            'test_data': test_data,
            'curve_data': curve_data,
            'train_error': train_error,
            'test_error': test_error
        })
    
    return plot_data

if __name__ == "__main__":
    # 示例用法
    n_samples_list = [10, 20, 50, 100, 200]
    degree = 3
    
    # 生成学习曲线数据
    learning_curve_data = generate_learning_curve_data(n_samples_list, degree)
    print("学习曲线数据：")
    for n_samples, train_error, test_error in zip(
        learning_curve_data['n_samples'],
        learning_curve_data['train_errors'],
        learning_curve_data['test_errors']
    ):
        print(f"样本数: {n_samples}, 训练误差: {train_error:.4f}, 测试误差: {test_error:.4f}")
    
    # 生成对比图数据
    comparison_plots = generate_comparison_plots(n_samples_list, degree)
    print("\n对比图数据已生成") 