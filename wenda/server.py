from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import os

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSHTTPRequestHandler)
    print(f"启动服务器在 http://localhost:{port}")
    print("按 Ctrl+C 停止服务器")
    try:
        webbrowser.open(f'http://localhost:{port}/index.html')
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务器")
        httpd.server_close()

if __name__ == '__main__':
    run_server()