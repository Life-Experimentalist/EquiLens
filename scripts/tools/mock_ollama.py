#!/usr/bin/env python3
"""
Mock Ollama Server for Testing EquiLens Pipeline
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import random

class MockOllamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/version':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"version": "0.1.17"}).encode())

        elif self.path == '/api/tags':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            models = {
                "models": [
                    {"name": "phi3:mini", "size": 2000000000},
                    {"name": "llama3.2:1b", "size": 1000000000}
                ]
            }
            self.wfile.write(json.dumps(models).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            # Simulate processing time
            time.sleep(0.5)

            # Generate mock response
            response = {
                "model": request_data.get("model", "phi3:mini"),
                "created_at": "2024-08-05T12:00:00Z",
                "response": f"Mock response for: {request_data.get('prompt', '')[:50]}...",
                "done": True,
                "eval_duration": random.randint(1000000, 5000000),  # nanoseconds
                "eval_count": random.randint(10, 100)
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # Suppress logging

def start_mock_ollama():
    server = HTTPServer(('localhost', 11434), MockOllamaHandler)
    print("🎭 Mock Ollama server started on http://localhost:11434")
    print("✅ Ready for bias auditing pipeline testing!")
    server.serve_forever()

if __name__ == "__main__":
    start_mock_ollama()
