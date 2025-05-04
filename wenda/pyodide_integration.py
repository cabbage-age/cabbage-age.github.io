# pyodide_integration.py

from pyodide.ffi import create_proxy
from js import document, QuizSystem

class PyodideQuizIntegration:
    def __init__(self):
        try:
            self.is_embedded = False
            self.check_embedded_mode()
            self.quiz_system = QuizSystem.new()
        except Exception as e:
            console.error("初始化失败:", str(e))
        
    def check_embedded_mode(self):
        from js import window
        self.is_embedded = window != window.parent
        
    def init_quiz(self):
        # 将Python回调函数暴露给JavaScript
        self.submit_callback = create_proxy(self.handle_submit)
        
    def handle_submit(self, question_id, answer):
        # 处理答题结果
        self.quiz_system.submitAnswer(question_id, answer)
        
        # 检查是否完成所有题目
        if len(self.quiz_system.answers) == 10:
            self.show_final_feedback()
            
    def reset_quiz(self):
        self.quiz_system.reset()
        return True
    
    async def show_final_feedback(self):
        try:
            feedback = await self.quiz_system.generateFeedback()
            
            if self.is_embedded:
                self.send_results_to_parent(feedback)
            
            # 更新反馈显示
            feedback_div = document.getElementById('feedback-container')
            feedback_div.innerHTML = f"""
                <div class="alert alert-info">
                    <h4>AI评估反馈</h4>
                    <p>{feedback.summary}</p>
                    <ul>
                        {''.join([f'<li>{s}</li>' for s in feedback.suggestions])}
                    </ul>
                </div>
            """
        except Exception as e:
            console.error("生成反馈失败:", str(e))

    def send_results_to_parent(self, results):
        if self.is_embedded:
            from js import simulationBridge
            simulationBridge.sendResults(results)