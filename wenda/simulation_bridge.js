class SimulationBridge {
    constructor() {
        this.isEmbedded = window !== window.parent;
        this.simulationData = {};
        this.isInitialized = false;
        this.pendingMessages = [];
    }

    init() {
        try {
            if (this.isEmbedded) {
                window.addEventListener('message', this.handleMessage.bind(this));
                this.notifyReady();
                this.isInitialized = true;
            }
        } catch (error) {
            console.error('初始化失败:', error);
        }
    }

    notifyReady() {
        window.parent.postMessage({
            type: 'quizSystemReady',
            status: 'ready'
        }, '*');
    }

    handleMessage(event) {
        if (!this.isInitialized) {
            this.pendingMessages.push(event);
            return;
        }
        const { type, data } = event.data;
        switch (type) {
            case 'startQuiz':
                this.startQuiz();
                break;
            case 'getResults':
                this.sendResults();
                break;
        }
    }

    startQuiz() {
        if (window.quizSystem) {
            window.quizSystem.reset();
            renderQuestion(0);
        }
    }

    sendResults(results) {
        if (this.isEmbedded) {
            window.parent.postMessage({
                type: 'quizResults',
                results: results
            }, '*');
        }
    }
}

// 全局初始化
window.simulationBridge = new SimulationBridge();
window.simulationBridge.init();
