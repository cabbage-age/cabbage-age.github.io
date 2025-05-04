const CONFIG = {
    // API配置
    XUNFEI_APP_ID: "30cf8364",
    XUNFEI_API_KEY: "8c30abd566c9bae992582cab8476b5b8", 
    XUNFEI_API_SECRET: "NTM3OWJkZGI4Yjc2NDQ0Y2FkODhjMWZl",
    XUNFEI_API_URL: "wss://spark-api.xf-yun.com/v4.0/chat",

    // AI模型配置
    MODEL: {
        VERSION: "Spark4.0 Ultra",
        TEMPERATURE: 0.7,
        MAX_TOKENS: 4096,
        TOP_K: 4
    },

    // 系统配置
    SYSTEM: {
        MAX_RETRIES: 3,
        RETRY_DELAY: 1000,
        REQUEST_TIMEOUT: 15000,
        AUTO_SAVE: true,
        ENABLE_VOICE: true
    },

    // UI配置
    UI: {
        THEME: "light",
        ANIMATION: true,
        LOADING_TEXT: "正在思考中...",
        SUCCESS_DURATION: 3000
    }
};

export default CONFIG;
