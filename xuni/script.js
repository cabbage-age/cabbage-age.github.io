let pyodide = null;
let plot = null;

async function initializePyodide() {
    // 显示加载状态
    const loadingStatus = document.getElementById('loading-status');
    const progressBar = document.querySelector('.progress');
    loadingStatus.style.display = 'block';
    
    try {
        // 初始化Pyodide
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
        });
        
        // 更新进度条
        progressBar.style.width = '20%';
        
        // 预加载必要的包
        const packages = [
            'numpy',
            'scipy',
            'scikit-learn',
            'pandas',
            'matplotlib'
        ];
        
        const totalPackages = packages.length;
        let loadedPackages = 0;
        
        for (const package of packages) {
            try {
                console.log(`正在加载包: ${package}`);
                await pyodide.loadPackage(package);
                loadedPackages++;
                progressBar.style.width = `${20 + (loadedPackages * 80 / totalPackages)}%`;
                console.log(`成功加载包: ${package}`);
            } catch (error) {
                console.error(`加载包 ${package} 失败:`, error);
            }
        }
        
        // 隐藏加载状态
        loadingStatus.style.display = 'none';
        
        // 初始化图表
        updatePlot();
    } catch (error) {
        console.error('初始化失败:', error);
        loadingStatus.innerHTML = `<p style="color: red;">加载失败: ${error.message}</p>`;
    }
}

function updatePlot() {
    if (!pyodide) {
        console.error('Pyodide未初始化');
        return;
    }
    
    const nSamples = parseInt(document.getElementById('n-samples').value);
    const noise = parseFloat(document.getElementById('noise').value);
    const degree = parseInt(document.getElementById('degree').value);
    const trainRatio = parseInt(document.getElementById('train-ratio').value) / 100;
    
    // 执行Python代码
    pyodide.runPythonAsync(`
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 生成数据
def generate_data(n_samples=${nSamples}, noise=0.1):
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y

# 多项式特征转换
def create_polynomial_features(X, degree):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

# 训练模型
def train_model(X_poly, y, train_idx):
    model = Ridge(alpha=0.0001)
    model.fit(X_poly[train_idx], y[train_idx])
    return model

# 计算误差
def calculate_errors(model, X_poly, y, train_idx, test_idx):
    train_error = mean_squared_error(y[train_idx], model.predict(X_poly[train_idx]))
    test_error = mean_squared_error(y[test_idx], model.predict(X_poly[test_idx]))
    return train_error, test_error

# 生成数据
X, y = generate_data(noise=${noise})
n_samples = len(X)

# 划分训练集和测试集
n_train = int(n_samples * ${trainRatio})
train_idx = np.random.choice(n_samples, n_train, replace=False)
test_idx = np.array([i for i in range(n_samples) if i not in train_idx])

# 创建多项式特征
X_poly = create_polynomial_features(X, degree=${degree})

# 训练模型
model = train_model(X_poly, y, train_idx)

# 计算误差
train_error, test_error = calculate_errors(model, X_poly, y, train_idx, test_idx)

# 生成预测曲线
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
X_plot_poly = create_polynomial_features(X_plot, degree=${degree})
y_plot = model.predict(X_plot_poly)

# 准备绘图数据
train_data = {
    'x': X[train_idx].flatten().tolist(),
    'y': y[train_idx].tolist(),
    'name': '训练数据',
    'mode': 'markers',
    'marker': {'color': 'blue', 'size': 8}
}

test_data = {
    'x': X[test_idx].flatten().tolist(),
    'y': y[test_idx].tolist(),
    'name': '测试数据',
    'mode': 'markers',
    'marker': {'color': 'red', 'size': 8}
}

curve_data = {
    'x': X_plot.flatten().tolist(),
    'y': y_plot.tolist(),
    'name': '拟合曲线',
    'mode': 'lines',
    'line': {'color': 'green', 'width': 2}
}

# 返回绘图数据
import json
json.dumps([train_data, test_data, curve_data, float(train_error), float(test_error)])
`).then(result => {
        try {
            // 解析JSON数据
            const [trainData, testData, curveData, trainError, testError] = JSON.parse(result);
            
            console.log('训练数据:', trainData);
            console.log('测试数据:', testData);
            console.log('曲线数据:', curveData);
            
            const layout = {
                title: {
                    text: `多项式拟合 (${degree}阶)<br>训练误差: ${trainError.toFixed(4)} | 测试误差: ${testError.toFixed(4)}`,
                    font: {
                        size: 14
                    }
                },
                xaxis: {
                    title: 'X',
                    range: [-3, 3],
                    gridcolor: '#f0f0f0'
                },
                yaxis: {
                    title: 'Y',
                    gridcolor: '#f0f0f0'
                },
                showlegend: true,
                legend: {
                    x: 1,
                    y: 1,
                    bgcolor: 'rgba(255, 255, 255, 0.8)'
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: {
                    l: 50,
                    r: 50,
                    t: 80,
                    b: 50
                }
            };
            
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };
            
            if (plot) {
                Plotly.newPlot('plot', [trainData, testData, curveData], layout, config);
            } else {
                plot = Plotly.newPlot('plot', [trainData, testData, curveData], layout, config);
            }
        } catch (error) {
            console.error('处理数据时出错:', error);
            alert('处理数据时出错: ' + error.message);
        }
    }).catch(error => {
        console.error('更新图表失败:', error);
        alert('更新图表时发生错误: ' + error.message);
    });
}

// 初始化滑块值显示
document.getElementById('n-samples').addEventListener('input', function(e) {
    document.getElementById('n-samples-value').textContent = e.target.value;
    updatePlot();
});

document.getElementById('noise').addEventListener('input', function(e) {
    document.getElementById('noise-value').textContent = e.target.value;
    updatePlot();
});

document.getElementById('degree').addEventListener('input', function(e) {
    document.getElementById('degree-value').textContent = e.target.value;
    updatePlot();
});

document.getElementById('train-ratio').addEventListener('input', function(e) {
    document.getElementById('train-ratio-value').textContent = e.target.value + '%';
    updatePlot();
});

// 高阶实验按钮点击事件
document.getElementById('advanced-experiment-btn').addEventListener('click', function() {
    if (!pyodide) {
        alert('Python环境还未初始化完成，请稍候再试');
        return;
    }
    
    // 先检查并加载必要的包
    const loadAdvancedPackages = async () => {
        try {
            // 显示加载状态
            const loadingStatus = document.getElementById('loading-status');
            const progressBar = document.querySelector('.progress');
            loadingStatus.style.display = 'block';
            progressBar.style.width = '50%';
            
            // 检查是否已加载所需包
            const hasPackages = await pyodide.runPythonAsync(`
try:
    import numpy
    import scipy
    from sklearn import datasets
    "success"
except ImportError:
    "failure"
`);
            
            if (hasPackages === "failure") {
                console.log("正在加载高阶实验所需的Python包...");
                
                const packages = [
                    'numpy',
                    'scipy',
                    'scikit-learn'
                ];
                
                for (const pkg of packages) {
                    await pyodide.loadPackage(pkg);
                    console.log(`已加载包: ${pkg}`);
                }
                
                console.log("高阶实验所需包加载完成");
            }
            
            // 隐藏加载状态
            loadingStatus.style.display = 'none';
            
            return true;
        } catch (error) {
            console.error("加载包时出错:", error);
            alert("加载Python包时出错: " + error.message);
            return false;
        }
    };
    
    // 加载包后再初始化界面
    loadAdvancedPackages().then(success => {
        if (!success) return;
        
        // 切换界面布局 - 使用简单编辑器，避免外部依赖
        const container = document.querySelector('.container');
        // 清除现有内容前先获取header
        const header = document.querySelector('.header');
        // 创建header的克隆副本，避免在清空容器时丢失引用
        const headerClone = header ? header.cloneNode(true) : null;
        container.innerHTML = '';
        if (headerClone) {
            container.appendChild(headerClone);
        }
        
        // 添加高阶实验内容
        const advancedContent = document.createElement('div');
        advancedContent.innerHTML = `
            <div class="code-editor-container">
                <div class="header-controls">
                    <button id="back-to-basic" class="btn-secondary">返回基础实验</button>
                    <div class="layout-controls">
                        <button id="toggle-layout" class="btn-secondary">
                            <i class="fas fa-columns"></i> 切换为垂直布局
                        </button>
                    </div>
                </div>
                
                <div class="advanced-layout horizontal-layout">
                    <div class="code-editor-panel">
                        <div class="editor-tabs">
                            <button class="editor-tab active" data-section="data-gen">数据生成</button>
                            <button class="editor-tab" data-section="feature-transform">特征变换</button>
                            <button class="editor-tab" data-section="model-train">模型训练</button>
                        </div>
                        <div class="editor-sections">
                            <div id="data-gen-section" class="editor-section active">
                                <h4>数据生成函数</h4>
                                <p class="description">修改此函数以生成不同的数据分布</p>
                                <textarea id="data-gen-editor" class="simple-editor" spellcheck="false" wrap="off"># 生成数据
def generate_data(n_samples=20, noise=0.2):
    # 使用说明：
    # - 可修改 n_samples 参数控制数据点数量
    # - 可修改 noise 参数控制噪声强度
    # - 可尝试修改数据生成函数，例如使用多项式函数生成数据
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y</textarea>
                            </div>
                            <div id="feature-transform-section" class="editor-section">
                                <h4>特征变换函数</h4>
                                <p class="description">修改特征变换方法以观察对拟合结果的影响</p>
                                <textarea id="feature-transform-editor" class="simple-editor" spellcheck="false" wrap="off"># 多项式特征转换
def create_polynomial_features(X, degree=3):
    # 使用说明：
    # - 修改 degree 参数可调整多项式阶数（值越大越容易过拟合）
    # - 可以尝试其他特征变换方法，如：
    #   - 对数变换: np.log1p(X)
    #   - 指数变换: np.exp(X)
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)</textarea>
                            </div>
                            <div id="model-train-section" class="editor-section">
                                <h4>模型训练函数</h4>
                                <p class="description">修改模型训练方法和参数</p>
                                <textarea id="model-train-editor" class="simple-editor" spellcheck="false" wrap="off"># 训练模型
def train_model(X_poly, y, train_idx):
    # 使用说明：
    # - 正则化说明:
    #   1. L2正则化(Ridge回归): 默认使用L2正则化防止过拟合
    #      alpha参数控制L2正则化强度，值越大约束越强
    #   2. L1正则化(Lasso回归): 将模型改为 model = Lasso(alpha=0.01)
    #      需要先导入: from sklearn.linear_model import Lasso
    #   3. 同时使用L1和L2(ElasticNet): model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    #      需要先导入: from sklearn.linear_model import ElasticNet
    model = Ridge(alpha=0.0001)  # L2正则化，alpha参数控制正则化强度
    model.fit(X_poly[train_idx], y[train_idx])
    return model

# 计算误差
def calculate_errors(model, X_poly, y, train_idx, test_idx):
    # 使用说明：
    # - 默认使用均方误差(MSE)评估模型性能
    # - 可以尝试使用其他评估指标，如平均绝对误差(MAE)
    #   from sklearn.metrics import mean_absolute_error
    #   train_error = mean_absolute_error(y[train_idx], model.predict(X_poly[train_idx]))
    train_error = mean_squared_error(y[train_idx], model.predict(X_poly[train_idx]))
    test_error = mean_squared_error(y[test_idx], model.predict(X_poly[test_idx]))
    return train_error, test_error</textarea>
                            </div>
                        </div>
                        <div class="code-buttons">
                            <button id="run-code" class="btn-primary">运行代码</button>
                            <button id="reset-code" class="btn-secondary">重置代码</button>
                            <button id="auto-indent" class="btn-secondary">格式化代码</button>
                        </div>
                    </div>
                    
                    <div class="result-panel">
                        <div class="plot-container advanced-plot">
                            <div id="plot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(advancedContent);
        
        // 绑定事件
        bindAdvancedExperimentEvents();
        
        // 更新图表
        updateAdvancedPlot();
    });
});

// 高阶实验中的事件绑定
function bindAdvancedExperimentEvents() {
    // 返回基础实验按钮
    document.getElementById('back-to-basic').addEventListener('click', function() {
        location.reload(); // 刷新页面返回基础实验
    });
    
    // 切换布局按钮
    document.getElementById('toggle-layout').addEventListener('click', function() {
        const advancedLayout = document.querySelector('.advanced-layout');
        advancedLayout.classList.toggle('horizontal-layout');
        
        // 更新按钮文本
        const isHorizontal = advancedLayout.classList.contains('horizontal-layout');
        this.innerHTML = isHorizontal 
            ? '<i class="fas fa-columns"></i> 切换为垂直布局' 
            : '<i class="fas fa-columns"></i> 切换为横向布局';
        
        // 延迟一小段时间后让图表重新绘制以适应新布局
        setTimeout(() => {
            if (Plotly && document.getElementById('plot')) {
                Plotly.relayout('plot', {
                    autosize: true
                });
            }
        }, 300);
    });
    
    // 编辑器标签切换
    document.querySelectorAll('.editor-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            // 移除所有active类
            document.querySelectorAll('.editor-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.editor-section').forEach(s => s.classList.remove('active'));
            
            // 添加active类到当前标签和对应区域
            this.classList.add('active');
            const sectionId = this.getAttribute('data-section') + '-section';
            document.getElementById(sectionId).classList.add('active');
        });
    });
    
    // 运行代码按钮
    document.getElementById('run-code').addEventListener('click', function() {
        updateAdvancedPlot();
    });
    
    // 重置代码按钮
    document.getElementById('reset-code').addEventListener('click', function() {
        document.getElementById('data-gen-editor').value = `# 生成数据
def generate_data(n_samples=20, noise=0.2):
    # 使用说明：
    # - 可修改 n_samples 参数控制数据点数量
    # - 可修改 noise 参数控制噪声强度
    # - 可尝试修改数据生成函数，例如使用多项式函数生成数据
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y`;
        
        document.getElementById('feature-transform-editor').value = `# 多项式特征转换
def create_polynomial_features(X, degree=3):
    # 使用说明：
    # - 修改 degree 参数可调整多项式阶数（值越大越容易过拟合）
    # - 可以尝试其他特征变换方法，如：
    #   - 对数变换: np.log1p(X)
    #   - 指数变换: np.exp(X)
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)`;
        
        document.getElementById('model-train-editor').value = `# 训练模型
def train_model(X_poly, y, train_idx):
    # 使用说明：
    # - 正则化说明:
    #   1. L2正则化(Ridge回归): 默认使用L2正则化防止过拟合
    #      alpha参数控制L2正则化强度，值越大约束越强
    #   2. L1正则化(Lasso回归): 将模型改为 model = Lasso(alpha=0.01)
    #      需要先导入: from sklearn.linear_model import Lasso
    #   3. 同时使用L1和L2(ElasticNet): model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    #      需要先导入: from sklearn.linear_model import ElasticNet
    model = Ridge(alpha=0.0001)  # L2正则化，alpha参数控制正则化强度
    model.fit(X_poly[train_idx], y[train_idx])
    return model

# 计算误差
def calculate_errors(model, X_poly, y, train_idx, test_idx):
    # 使用说明：
    # - 默认使用均方误差(MSE)评估模型性能
    # - 可以尝试使用其他评估指标，如平均绝对误差(MAE)
    #   from sklearn.metrics import mean_absolute_error
    #   train_error = mean_absolute_error(y[train_idx], model.predict(X_poly[train_idx]))
    train_error = mean_squared_error(y[train_idx], model.predict(X_poly[train_idx]))
    test_error = mean_squared_error(y[test_idx], model.predict(X_poly[test_idx]))
    return train_error, test_error`;
    });
    
    // 简单格式化代码按钮（只处理缩进）
    document.getElementById('auto-indent').addEventListener('click', function() {
        formatActiveEditor();
    });
}

// 简单的代码格式化函数
function formatActiveEditor() {
    // 获取当前活动的编辑器
    const activeSection = document.querySelector('.editor-section.active');
    if (!activeSection) return;
    
    const editorId = activeSection.querySelector('textarea').id;
    const editor = document.getElementById(editorId);
    if (!editor) return;
    
    const code = editor.value;
    const lines = code.split('\n');
    let formattedLines = [];
    let indentLevel = 0;
    
    for (let line of lines) {
        // 去除行首尾空格
        const trimmedLine = line.trim();
        
        // 空行保持原样
        if (trimmedLine === '') {
            formattedLines.push('');
            continue;
        }
        
        // 检查是否需要减少缩进（如果行以特定关键字开头）
        if (/^(else|elif|except|finally)/.test(trimmedLine)) {
            indentLevel = Math.max(0, indentLevel - 1);
        }
        
        // 添加缩进后的行
        formattedLines.push(' '.repeat(4 * indentLevel) + trimmedLine);
        
        // 检查是否需要增加缩进（如果行以冒号结尾）
        if (trimmedLine.endsWith(':')) {
            indentLevel += 1;
        }
    }
    
    editor.value = formattedLines.join('\n');
}

// 高阶实验的图表更新函数
function updateAdvancedPlot() {
    if (!pyodide) {
        console.error('Pyodide未初始化');
        return;
    }
    
    // 确保所有必要的包已加载
    const loadPackages = async () => {
        try {
            // 检查是否已加载numpy和scikit-learn
            const checkResult = await pyodide.runPythonAsync(`
try:
    # 尝试导入必要的包
    import numpy
    import scipy
    from sklearn import preprocessing
    "success"
except ImportError:
    "failure"
            `);
            
            if (checkResult === "failure") {
                console.log("正在加载必要的Python包...");
                await pyodide.loadPackage(["numpy", "scipy", "scikit-learn"]);
                console.log("Python包加载完成");
            }
            
            return true;
        } catch (error) {
            console.error("加载包时出错:", error);
            alert("加载Python包时出错: " + error.message);
            return false;
        }
    };
    
    const trainRatio = 0.7; // 默认训练集比例为70%
    
    // 从简单编辑器获取代码
    const dataGenCode = document.getElementById('data-gen-editor').value;
    const featureTransformCode = document.getElementById('feature-transform-editor').value;
    const modelTrainCode = document.getElementById('model-train-editor').value;
    
    const fullCode = `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

${dataGenCode}

${featureTransformCode}

${modelTrainCode}`;
    
    // 先确保包加载，然后执行代码
    loadPackages().then(success => {
        if (!success) return;
        
        // 将用户代码和执行逻辑分开
        // 首先将用户代码保存到全局变量
        pyodide.runPythonAsync(`
user_code = """
${fullCode}
"""
        `).then(() => {
            // 然后执行我们的处理代码，分开执行避免嵌套try-except导致的语法错误
            pyodide.runPythonAsync(`
import json
import traceback

# 自定义异常处理
try:
    # 执行用户代码
    exec(user_code)
    
    # 生成数据
    X, y = generate_data()
    n_samples = len(X)

    # 划分训练集和测试集
    n_train = int(n_samples * ${trainRatio})
    train_idx = np.random.choice(n_samples, n_train, replace=False)
    test_idx = np.array([i for i in range(n_samples) if i not in train_idx])

    # 确保degree变量存在
    degree_value = 3  # 默认值
    try:
        degree_value = degree  # 尝试获取用户定义的值
    except NameError:
        pass  # 如果没有定义，使用默认值

    # 创建多项式特征
    X_poly = create_polynomial_features(X, degree_value)

    # 训练模型
    model = train_model(X_poly, y, train_idx)

    # 计算误差
    train_error, test_error = calculate_errors(model, X_poly, y, train_idx, test_idx)
    
    # 计算过拟合比率
    overfit_ratio = test_error / train_error if train_error > 0 else 0

    # 生成预测曲线
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_plot_poly = create_polynomial_features(X_plot, degree_value)
    y_plot = model.predict(X_plot_poly)

    # 准备绘图数据
    train_data = {
        'x': X[train_idx].flatten().tolist(),
        'y': y[train_idx].tolist(),
        'name': '训练数据',
        'mode': 'markers',
        'marker': {'color': 'blue', 'size': 8}
    }

    test_data = {
        'x': X[test_idx].flatten().tolist(),
        'y': y[test_idx].tolist(),
        'name': '测试数据',
        'mode': 'markers',
        'marker': {'color': 'red', 'size': 8}
    }

    curve_data = {
        'x': X_plot.flatten().tolist(),
        'y': y_plot.tolist(),
        'name': '拟合曲线',
        'mode': 'lines',
        'line': {'color': 'green', 'width': 2}
    }

    result = {
        'status': 'success',
        'data': [train_data, test_data, curve_data, float(train_error), float(test_error)],
        'degree': int(degree_value),
        'overfit_ratio': float(overfit_ratio)
    }
    
except Exception as e:
    result = {
        'status': 'error',
        'message': str(e),
        'traceback': traceback.format_exc()
    }

# 返回结果
json.dumps(result)
            `).then(result => {
                try {
                    // 解析JSON数据
                    const resultObj = JSON.parse(result);
                    
                    if (resultObj.status === 'error') {
                        console.error('执行Python代码时出错:', resultObj.message);
                        console.error('错误详情:', resultObj.traceback);
                        alert('执行Python代码时出错: ' + resultObj.message);
                        return;
                    }
                    
                    const [trainData, testData, curveData, trainError, testError] = resultObj.data;
                    const degree = resultObj.degree || 3;
                    const overfitRatio = resultObj.overfit_ratio;
                    
                    const layout = {
                        title: {
                            text: `多项式拟合 (高阶实验)`,
                            font: {
                                size: 14
                            }
                        },
                        xaxis: {
                            title: 'X',
                            range: [-3, 3],
                            gridcolor: '#f0f0f0'
                        },
                        yaxis: {
                            title: 'Y',
                            gridcolor: '#f0f0f0'
                        },
                        showlegend: true,
                        legend: {
                            x: 1,
                            y: 1,
                            bgcolor: 'rgba(255, 255, 255, 0.8)'
                        },
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white',
                        margin: {
                            l: 50,
                            r: 50,
                            t: 80,
                            b: 50
                        },
                        autosize: true
                    };
                    
                    const config = {
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false
                    };
                    
                    // 在绘制前检查当前布局状态，调整图表的细节
                    const isHorizontal = document.querySelector('.advanced-layout').classList.contains('horizontal-layout');
                    if (!isHorizontal) {
                        // 垂直布局下优化图表
                        layout.height = 550; // 设置更大的高度
                        layout.legend.y = 0.95; // 调整图例位置
                    }
                    
                    Plotly.newPlot('plot', [trainData, testData, curveData], layout, config);
                } catch (error) {
                    console.error('处理数据时出错:', error);
                    alert('处理代码时出错: ' + error.message);
                }
            }).catch(error => {
                console.error('执行代码时出错:', error);
                alert('执行代码时出错: ' + error.message);
            });
        }).catch(error => {
            console.error('保存用户代码时出错:', error);
            alert('保存用户代码时出错: ' + error.message);
        });
    });
}

// 页面加载完成后初始化Pyodide
window.addEventListener('load', initializePyodide);

function updateLearningCurve() {
    const degree = parseInt(document.getElementById('degree').value);
    const noise = parseFloat(document.getElementById('noise').value);
    
    // 发送请求获取学习曲线数据
    fetch('/api/learning_curve', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            degree: degree,
            noise: noise
        })
    })
    .then(response => response.json())
    .then(data => {
        const learningCurveTrace = {
            x: data.n_samples,
            y: data.train_errors,
            name: '训练误差',
            type: 'scatter',
            mode: 'lines+markers',
            line: {color: 'blue'}
        };
        
        const testCurveTrace = {
            x: data.n_samples,
            y: data.test_errors,
            name: '测试误差',
            type: 'scatter',
            mode: 'lines+markers',
            line: {color: 'red'}
        };
        
        const layout = {
            title: '学习曲线',
            xaxis: {title: '训练样本数量'},
            yaxis: {title: '均方误差'},
            showlegend: true
        };
        
        Plotly.newPlot('learning-curve-plot', [learningCurveTrace, testCurveTrace], layout);
    });
}

function updateComparisonPlots() {
    const degree = parseInt(document.getElementById('degree').value);
    const noise = parseFloat(document.getElementById('noise').value);
    
    // 发送请求获取对比图数据
    fetch('/api/comparison_plots', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            degree: degree,
            noise: noise
        })
    })
    .then(response => response.json())
    .then(data => {
        const container = document.getElementById('comparison-plots');
        container.innerHTML = '';
        
        data.forEach(plotData => {
            const plotDiv = document.createElement('div');
            plotDiv.className = 'comparison-plot';
            container.appendChild(plotDiv);
            
            const traces = [
                {
                    x: plotData.train_data.x,
                    y: plotData.train_data.y,
                    name: '训练数据',
                    mode: 'markers',
                    marker: {color: 'blue'}
                },
                {
                    x: plotData.test_data.x,
                    y: plotData.test_data.y,
                    name: '测试数据',
                    mode: 'markers',
                    marker: {color: 'red'}
                },
                {
                    x: plotData.curve_data.x,
                    y: plotData.curve_data.y,
                    name: '拟合曲线',
                    mode: 'lines',
                    line: {color: 'green'}
                }
            ];
            
            const layout = {
                title: `样本数量: ${plotData.n_samples}`,
                xaxis: {title: 'X'},
                yaxis: {title: 'Y'},
                showlegend: true,
                annotations: [{
                    x: 0.5,
                    y: 1.1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `训练误差: ${plotData.train_error.toFixed(4)}, 测试误差: ${plotData.test_error.toFixed(4)}`,
                    showarrow: false
                }]
            };
            
            Plotly.newPlot(plotDiv, traces, layout);
        });
    });
}

// 在页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 检查是否存在controls元素
    const controlsElement = document.getElementById('controls');
    
    // 只有在controls元素存在时才添加按钮
    if (controlsElement) {
        // 添加学习曲线和对比图的更新按钮
        const updateLearningCurveBtn = document.createElement('button');
        updateLearningCurveBtn.textContent = '更新学习曲线';
        updateLearningCurveBtn.onclick = updateLearningCurve;
        controlsElement.appendChild(updateLearningCurveBtn);
        
        const updateComparisonBtn = document.createElement('button');
        updateComparisonBtn.textContent = '更新对比图';
        updateComparisonBtn.onclick = updateComparisonPlots;
        controlsElement.appendChild(updateComparisonBtn);
        
        // 添加学习曲线和对比图的容器
        const learningCurveDiv = document.createElement('div');
        learningCurveDiv.id = 'learning-curve-plot';
        document.body.appendChild(learningCurveDiv);
        
        const comparisonDiv = document.createElement('div');
        comparisonDiv.id = 'comparison-plots';
        document.body.appendChild(comparisonDiv);
    }
});

// 添加编辑器说明
(function() {
    // 页面加载完成后执行
    document.addEventListener('DOMContentLoaded', function() {
        // 定义每个编辑器的说明文本
        const editorInstructions = {
            dataGen: `# 生成数据
def generate_data(n_samples=20, noise=0.2):
    # 使用说明：
    # - 可修改 n_samples 参数控制数据点数量
    # - 可修改 noise 参数控制噪声强度
    # - 可尝试修改数据生成函数，例如使用多项式函数生成数据
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y`,
            
            featureTransform: `# 多项式特征转换
def create_polynomial_features(X, degree=3):
    # 使用说明：
    # - 修改 degree 参数可调整多项式阶数（值越大越容易过拟合）
    # - 可以尝试其他特征变换方法，如：
    #   - 对数变换: np.log1p(X)
    #   - 指数变换: np.exp(X)
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)`,
            
            modelTrain: `# 训练模型
def train_model(X_poly, y, train_idx):
    # 使用说明：
    # - 正则化说明:
    #   1. L2正则化(Ridge回归): 默认使用L2正则化防止过拟合
    #      alpha参数控制L2正则化强度，值越大约束越强
    #   2. L1正则化(Lasso回归): 将模型改为 model = Lasso(alpha=0.01)
    #      需要先导入: from sklearn.linear_model import Lasso
    #   3. 同时使用L1和L2(ElasticNet): model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    #      需要先导入: from sklearn.linear_model import ElasticNet
    model = Ridge(alpha=0.0001)  # L2正则化，alpha参数控制正则化强度
    model.fit(X_poly[train_idx], y[train_idx])
    return model

# 计算误差
def calculate_errors(model, X_poly, y, train_idx, test_idx):
    # 使用说明：
    # - 默认使用均方误差(MSE)评估模型性能
    # - 可以尝试使用其他评估指标，如平均绝对误差(MAE)
    #   from sklearn.metrics import mean_absolute_error
    #   train_error = mean_absolute_error(y[train_idx], model.predict(X_poly[train_idx]))
    train_error = mean_squared_error(y[train_idx], model.predict(X_poly[train_idx]))
    test_error = mean_squared_error(y[test_idx], model.predict(X_poly[test_idx]))
    return train_error, test_error`
        };
        
        // 创建一个函数来更新编辑器内容
        const updateEditors = function() {
            // 获取所有编辑器元素
            const dataGenEditor = document.getElementById('data-gen-editor');
            const featureTransformEditor = document.getElementById('feature-transform-editor');
            const modelTrainEditor = document.getElementById('model-train-editor');
            
            // 如果编辑器元素存在，则更新其内容
            if (dataGenEditor) {
                dataGenEditor.value = editorInstructions.dataGen;
            }
            
            if (featureTransformEditor) {
                featureTransformEditor.value = editorInstructions.featureTransform;
            }
            
            if (modelTrainEditor) {
                modelTrainEditor.value = editorInstructions.modelTrain;
            }
        };
        
        // 高级实验按钮添加点击事件，在点击后执行更新操作
        const advancedBtn = document.getElementById('advanced-experiment-btn');
        if (advancedBtn) {
            const originalClick = advancedBtn.onclick;
            advancedBtn.onclick = function(e) {
                if (originalClick) {
                    originalClick.call(this, e);
                }
                
                // 等待DOM更新后更新编辑器内容
                setTimeout(updateEditors, 500);
            };
        }
    });
})(); 