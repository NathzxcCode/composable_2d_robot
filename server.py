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


class SimulatorState:
    def __init__(self):
        self.limbs = []
        self.connections = []
        self.pose_history = []
        self.max_history = 100
        self.last_update = None

    def update(self, data):
        self.limbs = data.get('limbs', [])
        self.connections = data.get('connections', [])
        self.last_update = time.time()
        
        pose_entry = {
            'timestamp': self.last_update,
            'limbs': self.limbs,
            'connections': self.connections
        }
        self.pose_history.append(pose_entry)
        
        if len(self.pose_history) > self.max_history:
            self.pose_history = self.pose_history[-self.max_history:]

    def get_simulation_state(self):
        return {
            'limbs': self.limbs,
            'connections': self.connections,
            'limb_count': len(self.limbs),
            'connection_count': len(self.connections),
            'last_update': self.last_update
        }

    def get_limb_poses(self):
        return {
            'limbs': self.limbs,
            'count': len(self.limbs)
        }

    def get_connections(self):
        return {
            'connections': self.connections,
            'count': len(self.connections)
        }

    def get_history(self):
        return {
            'poses': self.pose_history,
            'count': len(self.pose_history)
        }

    def clear_history(self):
        self.pose_history = []


simulator_state = SimulatorState()


class RobotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/simulation_state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(simulator_state.get_simulation_state()).encode())
            return
        
        if path == '/api/limb_poses':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(simulator_state.get_limb_poses()).encode())
            return
        
        if path == '/api/connections':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(simulator_state.get_connections()).encode())
            return
        
        if path == '/api/history':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(simulator_state.get_history()).encode())
            return
        
        if path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok',
                'has_data': len(simulator_state.limbs) > 0,
                'last_update': simulator_state.last_update
            }).encode())
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
                simulator_state.update(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'ok',
                    'limb_count': len(simulator_state.limbs),
                    'connection_count': len(simulator_state.connections)
                })
                self.wfile.write(response.encode())
            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        
        elif path == '/api/clear':
            simulator_state.clear_history()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'cleared'}).encode())
        
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