#!/usr/bin/env python3
"""
Simple HTTP Server for Robot Simulator
Provides API endpoints for GBP implementation to query simulation data

Architecture:
- Browser runs simulation and sends data to server (push)
- Server stores current state
- GBP client queries server for data (pull)

Usage: python3 server.py
Then open http://localhost:5000 in browser
"""

import http.server
import socketserver
import json
import os
import time
from urllib.parse import urlparse, parse_qs

PORT = 5000
DATA_FILE = "pose_data.json"

# Global list to store poses
pose_history = []

def save_pose_to_file():
    """Save pose history to JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(pose_history, f, indent=2)

class RobotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/simulation_state' or path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'has_data': True}).encode())
            return
        
        if path == '/' or path == '/index.html' or path == '/robot_simulator.html':
            self.path = '/robot_simulator.html'
        
        return super().do_GET()
    
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/update':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(body)
                pose_history.append(data)
                save_pose_to_file()
                print(f"Stored pose {len(pose_history)}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'ok'
                })
                self.wfile.write(response.encode())
            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

socketserver.TCPServer.allow_reuse_address = True

print("=" * 50)
print("Robot Simulator Server")
print("=" * 50)
print(f"Server running at: http://localhost:{PORT}")
print()
print("API Endpoints (for GBP client to pull data):")
print(f"  GET  /api/simulation_state - Get all limbs and connections")
print(f"  GET  /api/limb_poses       - Get limb poses with positions/angles")
print(f"  GET  /api/connections      - Get connection tree with calibrations")
print(f"  GET  /api/history          - Get pose history")
print(f"  GET  /api/health           - Check if server has data")
print()
print("For browser (push simulation data to server):")
print(f"  POST /api/update           - Send simulation state")
print(f"  POST /api/clear            - Clear pose history")
print("=" * 50)

with socketserver.TCPServer(("", PORT), RobotHandler) as httpd:
    httpd.serve_forever()