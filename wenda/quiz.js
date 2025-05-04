import CONFIG from './config.js';

// 在文件开头添加
export { CONFIG };  // 如果其他模块需要使用

const CryptoJS = window.CryptoJS;

export const questions = [
    {
        id: 1,
        question: "什么是过拟合的主要表现？",
        options: [
            "训练集表现很好，测试集表现差",
            "训练集和测试集都表现差",
            "训练集和测试集都表现好",
            "只在测试集表现差"
        ],
        correct: 0,
        category: "基本概念"
    },
    {
        id: 2,
        question: "下列哪个现象不是过拟合的特征？",
        options: [
            "模型复杂度高",
            "对训练数据中的噪声过度学习",
            "模型泛化能力强",
            "训练误差远小于测试误差"
        ],
        correct: 2,
        category: "基本概念"
    },
    {
        id: 3,
        question: "在多项式回归中，以下哪种情况最容易导致过拟合？",
        options: [
            "使用一次多项式",
            "使用二次多项式",
            "使用三次多项式",
            "使用十次多项式"
        ],
        correct: 3,
        category: "模型复杂度"
    },
    {
        id: 4,
        question: "要避免过拟合，下列哪种方法是不正确的？",
        options: [
            "增加训练数据量",
            "使用正则化",
            "增加模型复杂度",
            "使用交叉验证"
        ],
        correct: 2,
        category: "模型复杂度"
    },
    {
        id: 5,
        question: "在数据量较小的情况下，以下哪种做法最容易导致过拟合？",
        options: [
            "使用简单的线性模型",
            "使用较强的正则化",
            "使用复杂的深度神经网络",
            "使用集成学习方法"
        ],
        correct: 2,
        category: "数据量"
    },
    {
        id: 6,
        question: "关于训练数据量和过拟合的关系，下列说法正确的是？",
        options: [
            "训练数据量越大，越容易过拟合",
            "训练数据量越小，越容易过拟合",
            "训练数据量与过拟合没有关系",
            "只要有足够的训练轮次，数据量大小不重要"
        ],
        correct: 1,
        category: "数据量"
    },
    {
        id: 7,
        question: "以下哪种正则化方法最适合用来防止过拟合？",
        options: [
            "增加模型参数",
            "L1正则化",
            "增加学习率",
            "减少训练轮次"
        ],
        correct: 1,
        category: "正则化"
    },
    {
        id: 8,
        question: "Dropout技术可以防止过拟合的原因是？",
        options: [
            "增加了模型复杂度",
            "减少了神经元之间的依赖关系",
            "增加了训练数据量",
            "提高了学习率"
        ],
        correct: 1,
        category: "正则化"
    },
    {
        id: 9,
        question: "使用交叉验证可以帮助防止过拟合的原因是？",
        options: [
            "增加了模型复杂度",
            "减少了训练数据量",
            "可以更好地评估模型泛化能力",
            "加快了训练速度"
        ],
        correct: 2,
        category: "验证方法"
    },
    {
        id: 10,
        question: "早停（Early Stopping）技术防止过拟合的原理是？",
        options: [
            "在验证集性能开始下降时停止训练",
            "在训练集性能达到最好时停止训练",
            "在训练一定轮次后停止训练",
            "在学习率降到某个值时停止训练"
        ],
        correct: 0,
        category: "验证方法"
    }
];

// 添加状态管理
const QuizState = {
    INITIAL: 'initial',
    ANSWERING: 'answering',
    SHOWING_EXPLANATION: 'showing_explanation',
    COMPLETED: 'completed'
};

export class QuizSystem {
    constructor() {
        this.questions = questions; // 使用导入的 questions 数组
        this.currentQuestionIndex = 0;
        this.answers = [];
        this.categories = {
            "基本概念": 0,
            "模型复杂度": 0,
            "数据量": 0,
            "正则化": 0,
            "验证方法": 0
        };
        this.maxScorePerCategory = 2;
        this.startTime = Date.now();
        this.state = QuizState.INITIAL;
    }

    submitAnswer(questionId, answer) {
        try {
            if (typeof questionId !== 'number' || typeof answer !== 'number') {
                throw new Error('无效的参数类型');
            }
            const now = Date.now();
            const question = questions[questionId - 1];
            if (!question) {
                throw new Error('Invalid question ID');
            }

            const isCorrect = question.correct === answer;
            
            this.answers.push({
                questionId,
                answer,
                correct: isCorrect,
                timeSpent: now - this.startTime
            });

            if (isCorrect) {
                this.categories[question.category]++;
            }

            // 更新当前题目
            this.currentQuestionIndex = questionId - 1;
            
            // 自动保存
            if (window.featureManager && window.featureManager.features && window.featureManager.features.autoSave) {
                window.featureManager.saveProgress(this.exportResults());
            }

            return isCorrect;
        } catch (error) {
            console.error('提交答案时出错:', error);
            throw error;
        }
    }

    getCategoryScores() {
        const scores = {};
        for (let category in this.categories) {
            scores[category] = (this.categories[category] / this.maxScorePerCategory) * 100;
        }
        return scores;
    }

    getWeakCategories() {
        const scores = this.getCategoryScores();
        return Object.entries(scores)
            .filter(([_, score]) => score < 50)
            .map(([category]) => category);
    }

    async generateFeedback() {
        try {
            const weakCategories = this.getWeakCategories();
            const totalCorrect = this.answers.filter(a => a.correct).length;
            const overallScore = (totalCorrect / questions.length) * 100;

            let summary = "";
            let suggestions = [];

            // 根据得分情况生成反馈
            if (overallScore >= 80) {
                summary = "你对过拟合的理解已经很好了！";
            } else if (overallScore >= 60) {
                summary = "你对过拟合有基本的理解，但还需要加强一些方面。";
            } else {
                summary = "你需要加强对过拟合概念的整体理解。";
            }

            // 使用 AI 为每个薄弱类别生成建议
            if (weakCategories.length > 0) {
                try {
                    const prompt = `作为一个机器学习专家，请针对学习者在以下过拟合相关知识类别的薄弱项给出具体的学习建议（每个类别30字以内）：${weakCategories.join('、')}`;
                    
                    const response = await fetch(CONFIG.AI_API_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${CONFIG.AI_API_KEY}`
                        },
                        body: JSON.stringify({
                            model: "gpt-3.5-turbo",
                            messages: [{
                                role: "user",
                                content: prompt
                            }],
                            max_tokens: 200,
                            temperature: 0.7
                        })
                    });

                    if (!response.ok) {
                        throw new Error('AI API 请求失败');
                    }

                    const data = await response.json();
                    const aiSuggestions = data.choices[0].message.content.trim();
                    
                    // 将 AI 建议按类别分割并添加到建议列表
                    suggestions = aiSuggestions.split('\n').filter(s => s.trim());
                } catch (error) {
                    console.error('获取AI建议失败:', error);
                    // 如果 AI 建议获取失败，使用默认建议
                    weakCategories.forEach(category => {
                        switch (category) {
                            case "基本概念":
                                suggestions.push("建议重新学习过拟合的基本定义和特征，可以通过可视化工具加深理解。");
                                break;
                            case "模型复杂度":
                                suggestions.push("建议通过实践调整不同的模型复杂度（如多项式阶数），观察其对过拟合的影响。");
                                break;
                            case "数据量":
                                suggestions.push("建议尝试使用不同大小的数据集进行训练，理解数据量与过拟合的关系。");
                                break;
                            case "正则化":
                                suggestions.push("建议学习并实践不同的正则化方法（L1/L2正则化、Dropout等），理解它们防止过拟合的原理。");
                                break;
                            case "验证方法":
                                suggestions.push("建议深入学习交叉验证和早停等验证方法，理解它们在防止过拟合中的作用。");
                                break;
                        }
                    });
                }
            }

            return {
                summary,
                suggestions,
                scores: this.getCategoryScores(),
                overallScore
            };
        } catch (error) {
            console.error('生成反馈失败:', error);
            return {
                summary: "生成反馈时发生错误",
                suggestions: [],
                scores: this.getCategoryScores(),
                overallScore: 0
            };
        }
    }

    reset() {
        this.answers = [];
        this.categories = {
            "基本概念": 0,
            "模型复杂度": 0,
            "数据量": 0,
            "正则化": 0,
            "验证方法": 0
        };
        this.currentQuestionIndex = 0;
    }

    exportResults() {
        return {
            answers: this.answers,
            categories: this.categories,
            scores: this.getCategoryScores(),
            currentQuestion: this.currentQuestionIndex + 1,
            timestamp: new Date().toISOString()
        };
    }

    setState(newState) {
        this.state = newState;
        // 可以在这里添加状态变化的回调
    }

    canSubmitAnswer() {
        return this.state === QuizState.INITIAL || this.state === QuizState.ANSWERING;
    }
}

// 添加更新进度条的函数
function updateProgressBar(currentIndex, totalQuestions) {
    const progressBar = document.querySelector('.progress-bar');
    const progressNumbers = document.querySelector('.progress-numbers');
    const percentage = ((currentIndex + 1) / totalQuestions) * 100;
    
    progressBar.style.width = `${percentage}%`;
    progressNumbers.textContent = `${currentIndex + 1}/${totalQuestions}`;
}

export const renderQuestion = async (index, quizSystem, aiExplainer) => {
    try {
        console.log(`[renderQuestion] 开始渲染题目 ${index + 1}`);
        
        // 更新进度条
        updateProgressBar(index, quizSystem.questions.length);
        
        // 验证参数
        if (!quizSystem || !aiExplainer) {
            throw new Error('系统未完全初始化');
        }

        // 获取题目容器
        const quizContainer = document.getElementById('quiz-container');
        if (!quizContainer) {
            throw new Error('找不到题目容器');
        }

        // 清除之前的解析内容
        const analysisContent = document.getElementById('current-analysis');
        if (analysisContent) {
            analysisContent.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-lightbulb"></i>
                    <p>请选择答案查看解析</p>
                </div>
            `;
        }

        // 清除之前的下一题按钮
        const existingNextButton = document.getElementById('nextButton');
        if (existingNextButton) {
            existingNextButton.remove();
        }

        // 清除之前的选项状态
        const existingOptions = document.querySelectorAll('.option-item');
        existingOptions.forEach(option => {
            option.classList.remove('selected', 'correct', 'incorrect');
            option.style.pointerEvents = 'auto';
        });

        // 获取当前题目
        const question = quizSystem.questions[index];
        if (!question) {
            throw new Error('题目不存在');
        }

        // 渲染题目内容
        const questionHtml = `
            <div class="question-content">
                <h3 class="question-title">问题 ${index + 1}: ${question.question}</h3>
                <div class="options-container">
                    ${question.options.map((option, i) => `
                        <div class="option-item" data-index="${i}">
                            <input type="radio" name="answer" id="option-${i}" value="${i}">
                            <label for="option-${i}">${option}</label>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        quizContainer.innerHTML = questionHtml;

        // 绑定选项点击事件
        const options = quizContainer.querySelectorAll('.option-item');
        options.forEach(option => {
            option.addEventListener('click', () => {
                const optionIndex = parseInt(option.dataset.index);
                selectOption(index, optionIndex, quizSystem, aiExplainer);
            });
        });

        // 更新状态
        quizSystem.state = QuizState.ANSWERING;
        console.log(`[renderQuestion] 题目 ${index + 1} 渲染完成`);

    } catch (error) {
        console.error('[renderQuestion] 渲染题目失败:', error);
        throw error;
    }
};

// 添加显示之前解析的函数
export async function showPreviousExplanation(questionIndex, answerRecord, aiExplainer) {
    const currentQuestion = questions[questionIndex];
    const selectedOptionText = currentQuestion.options[answerRecord.answer];
    
    // 显示解析
    const explanationDiv = document.createElement('div');
    explanationDiv.className = `alert alert-${answerRecord.correct ? 'success' : 'danger'} mt-3`;
    explanationDiv.innerHTML = `
        <h5>${answerRecord.correct ? '✓ 回答正确！' : '✗ 回答错误'}</h5>
        <div class="explanation-content">
            <p><strong>AI解析：</strong></p>
            <p>${await aiExplainer.getExplanation(
                currentQuestion.question,
                selectedOptionText,
                answerRecord.correct
            )}</p>
        </div>
    `;

    // 添加解析到当前题目卡片
    document.querySelector('.question-card').appendChild(explanationDiv);
}

export class AIExplainer {
    constructor(config) {
        if (!config || !config.XUNFEI_APP_ID || !config.XUNFEI_API_KEY || !config.XUNFEI_API_SECRET) {
            throw new Error('AIExplainer 初始化失败：缺少必要的讯飞API配置信息');
        }
        this.appId = config.XUNFEI_APP_ID;
        this.apiKey = config.XUNFEI_API_KEY;
        this.apiSecret = config.XUNFEI_API_SECRET;
        this.apiUrl = config.XUNFEI_API_URL;
        this.maxRetries = config.MAX_RETRIES || 3;
        this.retryDelay = config.RETRY_DELAY || 1000;
        console.log('AIExplainer初始化配置:', {
            appId: this.appId,
            apiKey: this.apiKey.substring(0, 4) + '****',
            apiUrl: this.apiUrl
        });
    }

    async getExplanation(question, selectedOption, isCorrect) {
        console.log('开始获取AI解析，参数:', {
            question,
            selectedOption,
            isCorrect
        });
        let retryCount = 0;
        
        while (retryCount < this.maxRetries) {
            try {
                console.log(`尝试获取AI解析 (第${retryCount + 1}次)`);
                
                return new Promise((resolve, reject) => {
                    let ws;
                    try {
                        const authUrl = this.getAuthUrl();
                        console.log('正在连接WebSocket...');
                        ws = new WebSocket(authUrl);
                    } catch (error) {
                        console.error('WebSocket 创建失败:', error);
                        reject(new Error('WebSocket 连接失败'));
                        return;
                    }

                    const timeout = setTimeout(() => {
                        console.log('WebSocket 请求超时');
                        ws.close();
                        reject(new Error('请求超时'));
                    }, 15000);

                    ws.onopen = () => {
                        console.log('WebSocket 连接成功，发送请求...');
                        const message = {
                            header: {
                                app_id: this.appId
                            },
                            parameter: {
                                chat: {
                                    domain: "4.0Ultra",
                                    temperature: 0.5,
                                    max_tokens: 4096,
                                    top_k: 4
                                }
                            },
                            payload: {
                                message: {
                                    text: [
                                        {
                                            role: "system",
                                            content: "你是一位机器学习专家，请用专业且简洁的语言解答问题。"
                                        },
                                        {
                                            role: "user",
                                            content: `请分析以下机器学习问题（100字以内）：

问题：${question}
选择答案：${selectedOption}
是否正确：${isCorrect ? '正确' : '错误'}

请按以下格式回答：
1. 答案分析：为什么对/错
2. 核心概念：相关知识点
3. 学习提示：如何理解`
                                        }
                                    ]
                                }
                            }
                        };

                        console.log('发送的消息:', JSON.stringify(message, null, 2));
                        ws.send(JSON.stringify(message));
                    };

                    let explanation = '';
                    ws.onmessage = (event) => {
                        try {
                            const response = JSON.parse(event.data);
                            console.log('收到WebSocket响应:', response);
                            
                            const code = response.header.code;
                            if (code !== 0) {
                                console.error('API返回错误:', response.header);
                                ws.close();
                                clearTimeout(timeout);
                                reject(new Error(response.header.message || '请求失败'));
                                return;
                            }
                            
                            if (response.payload && response.payload.choices) {
                                const content = response.payload.choices.text[0].content;
                                if (content) {
                                    explanation += content;
                                }
                                
                                // 当status为2时表示回答完成
                                if (response.payload.choices.status === 2) {
                                    ws.close();
                                    clearTimeout(timeout);
                                    resolve(explanation.trim() || '抱歉，未能获取到有效解析');
                                }
                            }
                        } catch (error) {
                            console.error('处理WebSocket消息失败:', error);
                            ws.close();
                            clearTimeout(timeout);
                            reject(error);
                        }
                    };

                    ws.onerror = (error) => {
                        console.error('WebSocket错误:', error);
                        clearTimeout(timeout);
                        ws.close();
                        reject(new Error('WebSocket连接错误'));
                    };

                    ws.onclose = () => {
                        console.log('WebSocket连接关闭');
                        clearTimeout(timeout);
                    };
                });

            } catch (error) {
                console.error(`AI解析尝试 ${retryCount + 1}/${this.maxRetries} 失败:`, error);
                
                if (retryCount === this.maxRetries - 1) {
                    console.log('使用默认解析');
                    return this.getDefaultExplanation(question, selectedOption, isCorrect);
                }
                
                retryCount++;
                console.log(`等待 ${this.retryDelay}ms 后重试...`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay));
            }
        }
        
        return this.getDefaultExplanation(question, selectedOption, isCorrect);
    }

    getAuthUrl() {
        try {
            const host = 'spark-api.xf-yun.com';
            const date = new Date().toUTCString();
            
            // 按照Python示例构建签名原文
            const signatureOrigin = `host: ${host}\ndate: ${date}\nGET /v4.0/chat HTTP/1.1`;
            
            console.log('签名原文:', signatureOrigin);
            
            // 使用 HMAC-SHA256 加密
            const signatureSha = CryptoJS.HmacSHA256(signatureOrigin, this.apiSecret);
            const signatureShaBase64 = CryptoJS.enc.Base64.stringify(signatureSha);
            
            // 构建鉴权字符串（与Python示例保持一致）
            const authorizationOrigin = `api_key="${this.apiKey}", algorithm="hmac-sha256", headers="host date request-line", signature="${signatureShaBase64}"`;
            const authorization = btoa(authorizationOrigin);
            
            // 构建URL参数
            const params = {
                authorization: authorization,
                date: date,
                host: host
            };
            
            // 构建最终URL
            const url = `${this.apiUrl}?${new URLSearchParams(params)}`;
            
            console.log('生成的鉴权URL:', url);
            return url;
        } catch (error) {
            console.error('生成鉴权URL失败:', error);
            throw new Error('生成鉴权URL失败: ' + error.message);
        }
    }

    getDefaultExplanation(question, selectedOption, isCorrect) {
        // 提供更详细的本地解析
        if (question.includes("过拟合的主要表现")) {
            return isCorrect ? 
                "正确！过拟合的典型特征就是模型在训练集上表现优秀，但在测试集上表现差，说明模型学习到了训练数据中的噪声。" :
                "过拟合最明显的特征是训练集和测试集性能的巨大差异，这表明模型没有真正学习到数据的本质特征。";
        }
        
        if (question.includes("不是过拟合的特征")) {
            return isCorrect ?
                "正确！模型具有强泛化能力恰恰说明没有过拟合，而是学习到了数据的本质特征。" :
                "模型泛化能力强意味着模型能够很好地应用到新数据上，这与过拟合的特征相反。";
        }
        
        if (question.includes("多项式回归")) {
            return isCorrect ?
                "正确！高次多项式（如十次）具有更强的拟合能力，更容易捕捉到训练数据中的噪声，导致过拟合。" :
                "多项式次数越高，模型复杂度越高，越容易发生过拟合。建议思考模型复杂度与过拟合的关系。";
        }
        
        if (question.includes("避免过拟合")) {
            return isCorrect ?
                "正确！增加模型复杂度反而会加剧过拟合，其他选项都是有效的防止过拟合的方法。" :
                "防止过拟合的关键是控制模型复杂度，增加模型复杂度反而会加剧过拟合问题。";
        }
        
        if (question.includes("数据量较小")) {
            return isCorrect ?
                "正确！数据量小时使用复杂的深度神经网络容易过拟合，因为模型参数远多于训练样本。" :
                "当数据量较小时，应选择简单的模型，避免使用过于复杂的模型结构。";
        }
        
        if (question.includes("训练数据量和过拟合的关系")) {
            return isCorrect ?
                "正确！数据量越小，模型越容易记住训练数据的特定模式，包括噪声，从而导致过拟合。" :
                "数据量与过拟合呈负相关，数据量越大，模型越不容易过拟合。";
        }
        
        if (question.includes("正则化方法")) {
            return isCorrect ?
                "正确！L1正则化通过引入惩罚项来限制模型复杂度，是防止过拟合的有效手段。" :
                "正则化的核心思想是通过添加惩罚项来限制模型复杂度，从而防止过拟合。";
        }
        
        if (question.includes("Dropout技术")) {
            return isCorrect ?
                "正确！Dropout通过随机失活神经元，减少神经元间的依赖，从而降低过拟合风险。" :
                "Dropout的主要作用是通过随机断开神经元连接，减少神经元之间的依赖，防止过拟合。";
        }
        
        if (question.includes("交叉验证")) {
            return isCorrect ?
                "正确！交叉验证通过多次划分训练集和验证集，能更准确地评估模型的泛化能力。" :
                "交叉验证的目的是更好地评估模型的泛化能力，从而及时发现过拟合问题。";
        }
        
        if (question.includes("早停")) {
            return isCorrect ?
                "正确！当验证集性能开始下降时及时停止训练，可以有效防止模型过拟合。" :
                "早停的原理是在模型开始过拟合（验证集性能下降）时及时停止训练。";
        }

        // 默认解析
        return isCorrect ?
            "回答正确！您很好地理解了这个概念。" :
            "建议重新思考这个概念，特别是它与过拟合的关系。";
    }

    generatePrompt(question, selectedOption, isCorrect) {
        return {
            role: "system",
            content: `你是一位经验丰富的机器学习专家和教育者。请用专业且生动的方式解答问题。

要求:
1. 语言风格要专业但通俗易懂
2. 解释要既有表面现象分析,也要有底层原理阐述
3. 要结合实际应用场景举例
4. 要引导学习者思考
5. 对错误选项要详细说明原因

问题: ${question}
学生选择: ${selectedOption}
是否正确: ${isCorrect ? '正确' : '错误'}

请按如下结构回答(限200字):
<div class="explanation-section">
    <p>
1. 答案分析：简明扼要说明对错原因，不要使用序号前缀
    </p>
</div>

<div class="explanation-section">
    <p>
2. 核心概念：解释相关的关键概念，不要使用序号前缀
    </p>
</div>

<div class="explanation-section">
    <p>
3. 学习提示：如何在实际工作中运用，不要使用序号前缀
    </p>
</div>

<div class="explanation-section">
    <p>
引导更深入的学习思考，不要使用序号前缀
    </p>
</div>`
        };
    }

    async getExplanationWithAnimation(question, selectedOption, isCorrect) {
        const explanation = await this.getExplanation(question, selectedOption, isCorrect);
        return this.animateText(explanation);
    }

    animateText(text, speed = 50) {
        return new Promise(resolve => {
            let result = '';
            const chars = text.split('');
            let i = 0;
            
            const interval = setInterval(() => {
                if (i < chars.length) {
                    result += chars[i];
                    i++;
                } else {
                    clearInterval(interval);
                    resolve(result);
                }
            }, speed);
        });
    }
}

export const selectOption = async (questionIndex, optionIndex, quizSystem, aiExplainer) => {
    try {
        console.log(`[selectOption] 选择题目 ${questionIndex + 1} 的选项 ${optionIndex}`);
        
        // 验证参数
        if (!quizSystem || !aiExplainer) {
            throw new Error('系统未完全初始化');
        }

        // 提交答案
        const isCorrect = quizSystem.submitAnswer(questionIndex + 1, optionIndex);
        
        // 更新选项样式
        const options = document.querySelectorAll('.option-item');
        options.forEach((option, i) => {
            option.classList.remove('selected', 'correct', 'incorrect');
            if (i === optionIndex) {
                option.classList.add('selected', isCorrect ? 'correct' : 'incorrect');
            }
        });

        // 禁用所有选项
        options.forEach(option => {
            option.style.pointerEvents = 'none';
        });

        // 显示解析
        const analysisContent = document.getElementById('current-analysis');
        if (analysisContent) {
            const explanation = await aiExplainer.getExplanation(
                quizSystem.questions[questionIndex].question,
                quizSystem.questions[questionIndex].options[optionIndex],
                isCorrect
            );
            
            analysisContent.innerHTML = `
                <div class="analysis-result">
                    <h4>${isCorrect ? '回答正确！' : '回答错误'}</h4>
                    <div class="explanation">${explanation}</div>
                </div>
            `;
        }

        // 检查是否已存在按钮
        const existingButton = document.getElementById('nextButton');
        if (!existingButton) {
            // 创建按钮
            const button = document.createElement('button');
            button.id = 'nextButton';
            button.className = 'btn btn-primary mt-3';
            
            // 判断是否是最后一道题
            const isLastQuestion = questionIndex === quizSystem.questions.length - 1;
            if (isLastQuestion) {
                button.textContent = '查看结果';
                button.addEventListener('click', async () => {
                    await showResults(quizSystem, aiExplainer);
                });
            } else {
                button.textContent = '下一题';
                button.addEventListener('click', async () => {
                    await nextQuestion(quizSystem, aiExplainer);
                });
            }

            const quizContainer = document.getElementById('quiz-container');
            if (quizContainer) {
                quizContainer.appendChild(button);
            }
        }

        // 更新状态
        quizSystem.state = QuizState.SHOWING_EXPLANATION;
        console.log(`[selectOption] 选项选择完成`);

    } catch (error) {
        console.error('[selectOption] 选择选项失败:', error);
        throw error;
    }
};

export const nextQuestion = async (quizSystem, aiExplainer) => {
    try {
        console.log('[nextQuestion] 开始切换到下一题');
        
        // 验证参数
        if (!quizSystem || !aiExplainer) {
            throw new Error('系统未完全初始化');
        }

        // 更新当前题目索引
        quizSystem.currentQuestionIndex++;
        
        // 渲染下一题
        await renderQuestion(quizSystem.currentQuestionIndex, quizSystem, aiExplainer);
        
        // 更新状态
        quizSystem.state = QuizState.ANSWERING;
        console.log('[nextQuestion] 切换到下一题完成');

    } catch (error) {
        console.error('[nextQuestion] 切换题目失败:', error);
        throw error;
    }
};

// 添加 showResults 函数
export async function showResults(quizSystem, aiExplainer) {
    try {
        // 隐藏题目区域
        const quizSection = document.querySelector('.quiz-section');
        const analysisSection = document.querySelector('.analysis-section');
        if (quizSection && analysisSection) {
            quizSection.style.display = 'none';
            analysisSection.style.display = 'none';
        }

        // 显示结果区域
        const resultSection = document.getElementById('result-section');
        if (resultSection) {
            resultSection.style.display = 'block';
            
            // 计算得分
            const scores = quizSystem.getCategoryScores();
            const feedback = await quizSystem.generateFeedback();
            
            // 更新结果内容
            resultSection.innerHTML = `
                <div class="card-body">
                    <h3 class="card-title">知识掌握情况分析</h3>
                    <div class="category-scores">
                        ${Object.entries(scores).map(([category, score]) => `
                            <div class="score-item">
                                <h5>${category}</h5>
                                <div class="progress">
                                    <div class="progress-bar" role="progressbar" style="width: ${score}%">
                                        ${score.toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="ai-feedback">
                        <h4>AI学习建议</h4>
                        <div class="feedback-content">
                            <p class="summary">${feedback.summary}</p>
                            ${feedback.suggestions.length > 0 ? `
                                <ul class="suggestions-list">
                                    ${feedback.suggestions.map(suggestion => `
                                        <li>${suggestion}</li>
                                    `).join('')}
                                </ul>
                            ` : ''}
                        </div>
                    </div>
                    <div class="mt-4 text-center">
                        <button class="btn btn-primary" onclick="window.location.reload()">
                            重新开始
                        </button>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('显示结果时出错:', error);
        showError('显示结果失败: ' + error.message);
    }
}

// 添加错误显示函数
function showError(message) {
    console.error('[showError] 显示错误消息:', message);
    const errorMessage = document.getElementById('error-message');
    if (errorMessage) {
        errorMessage.style.display = 'block';
        errorMessage.querySelector('p').textContent = message;
    } else {
        // 如果找不到错误消息元素，创建一个新的
        const errorDiv = document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.className = 'alert alert-danger';
        errorDiv.style.position = 'fixed';
        errorDiv.style.top = '20px';
        errorDiv.style.right = '20px';
        errorDiv.style.zIndex = '1000';
        errorDiv.innerHTML = `<p>${message}</p>`;
        document.body.appendChild(errorDiv);
        
        // 3秒后自动隐藏
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 3000);
    }
}

// 添加重试初始化函数
async function retryInitialization() {
    try {
        console.log('[retryInitialization] 开始重试初始化...');
        await initializeSystem();
    } catch (error) {
        console.error('[retryInitialization] 重试初始化失败:', error);
        showError('重试初始化失败: ' + error.message);
    }
}

// 确保在初始化时能获取到 CONFIG
async function initializeSystem() {
    try {
        console.log('[initializeSystem] 开始初始化系统...');
        
        // 1. 检查必要的依赖是否加载
        if (!window.CryptoJS) {
            throw new Error('CryptoJS 未加载');
        }
        
        // 检查 CONFIG 对象和必要的 key
        if (!window.CONFIG) {
            // 尝试从模块导入
            try {
                const configModule = await import('./config.js');
                window.CONFIG = configModule.default;
            } catch (error) {
                console.error('[initializeSystem] 导入配置文件失败:', error);
                throw new Error('无法加载配置文件');
            }
        }
        
        if (!window.CONFIG.XUNFEI_APP_ID) {
            throw new Error('配置信息不完整: 缺少讯飞API配置');
        }
        
        console.log('[initializeSystem] 配置信息检查通过');
        
        // 2. 初始化 FeatureManager (确保 FeatureManager 类已定义或导入)
        if (typeof FeatureManager !== 'undefined') {
             window.featureManager = new FeatureManager();
             console.log('[initializeSystem] FeatureManager 初始化完成');
        } else {
             console.warn('[initializeSystem] FeatureManager 未定义，跳过初始化');
             // 可能需要提供一个默认的空对象或实现，以防后续代码依赖它
             window.featureManager = { loadProgress: () => null, saveProgress: () => {} };
        }
        
        // 3. 初始化 QuizSystem 并检查是否有保存的进度
        window.quizSystem = new QuizSystem(); // 先创建实例
        const savedProgress = window.featureManager.loadProgress();
        let currentQuestionIndex = 0; // 默认从第一题开始

        if (savedProgress) {
            console.log('[initializeSystem] 检测到保存的进度:', savedProgress);
            // 添加更健壮的恢复逻辑，检查数据结构是否符合预期
            if (savedProgress.answers && savedProgress.categories && typeof savedProgress.currentQuestion === 'number') {
                 const continueQuiz = confirm('检测到上次的答题进度, 是否继续?');
                 if (continueQuiz) {
                     // 使用 Object.assign 恢复状态
                     Object.assign(window.quizSystem, savedProgress);
                     // *** 确保恢复的 currentQuestion 是索引 (从0开始) ***
                     // 如果保存的是题号 (从1开始)，需要减1
                     // 假设 savedProgress.currentQuestion 保存的是题号 (1-based)
                     currentQuestionIndex = savedProgress.currentQuestion > 0 ? savedProgress.currentQuestion - 1 : 0;
                     // 更新 quizSystem 的索引
                     window.quizSystem.currentQuestionIndex = currentQuestionIndex;
                     console.log(`[initializeSystem] 已恢复上次进度，将从题目索引 ${currentQuestionIndex} 开始`);
                 } else {
                     console.log('[initializeSystem] 用户选择不恢复进度，重新开始');
                 }
            } else {
                 console.warn('[initializeSystem] 保存的进度格式无效，将忽略并重新开始:', savedProgress);
            }
        } else {
            console.log('[initializeSystem] 未检测到保存的进度，从头开始');
        }
        console.log('[initializeSystem] QuizSystem 初始化完成');

        // 4. 初始化 AI 解析器（使用 window.CONFIG）
        try {
             window.aiExplainer = new AIExplainer(window.CONFIG);
             console.log('[initializeSystem] AIExplainer 初始化完成');
        } catch (aiError) {
             console.error('[initializeSystem] AIExplainer 初始化失败:', aiError);
             showError('AI解析器初始化失败: ' + aiError.message);
             // 可以考虑是否阻止后续流程，或者提供无 AI 解析的模式
             return; // 阻止继续渲染题目
        }

        // 5. 渲染第一题或继续上次的题目
        console.log(`[initializeSystem] 准备渲染题目，索引: ${currentQuestionIndex}`);
        await renderQuestion(currentQuestionIndex, window.quizSystem, window.aiExplainer);
        console.log(`[initializeSystem] 题目渲染调用完成`);

    } catch (error) {
        console.error('[initializeSystem] 系统初始化失败:', error);
        showError('系统初始化失败: ' + error.message);
        // 可以在这里添加更具体的错误处理，例如显示一个错误页面
    }
}

// *** 确保 initializeSystem 被调用 ***
// 通常在 HTML 文件末尾或使用 DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
     console.log('DOM 已加载，准备调用 initializeSystem');
     initializeSystem();
});

// 挂载 selectOption
window.selectOption = async (questionIndex, optionIndex) => {
    console.log(`[selectOption] 用户选择了题目 ${questionIndex + 1} 的选项 ${optionIndex}`);
    await selectOption(questionIndex, optionIndex, window.quizSystem, window.aiExplainer);
};