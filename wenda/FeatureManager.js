export class FeatureManager {
    constructor() {
        this.features = {
            autoSave: true,
            darkMode: false
        };
        this.setupAutoSave();
        this.initTheme();
    }

    // 自动保存进度
    saveProgress(data) {
        try {
            localStorage.setItem('quizProgress', JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('保存进度失败:', error);
            return false;
        }
    }

    // 加载保存的进度
    loadProgress() {
        try {
            const saved = localStorage.getItem('quizProgress');
            return saved ? JSON.parse(saved) : null;
        } catch (error) {
            console.error('加载进度失败:', error);
            return null;
        }
    }

    // 切换深色模式
    toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isDark);
        
        // 更新图标
        const themeIcon = document.querySelector('.theme-toggle i');
        themeIcon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }

    initTheme() {
        const isDark = localStorage.getItem('darkMode') === 'true';
        if (isDark) {
            document.body.classList.add('dark-mode');
        }
    }

    setupAutoSave() {
        if (this.features.autoSave) {
            // 每分钟自动保存一次
            this.autoSaveInterval = setInterval(() => {
                if (window.quizSystem) {
                    this.saveProgress(window.quizSystem.exportResults());
                }
            }, 60000);
        }
    }

    cleanup() {
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
        }
    }
} 