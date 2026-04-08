#!/usr/bin/env python3
"""
Client for interacting with the Robot Simulator API
Used by GBP implementation - polls server for new data

Architecture:
- Simulation pushes data to server (POST /api/update)
- This client polls server and runs GBP on each iteration

Usage: python3 client_example.py
       (or import in your GBP implementation)
"""

import requests
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

BASE_URL = "http://localhost:5000"


@dataclass
class LimbPose:
    """Represents a limb in the simulation"""
    id: int
    local_angle: float
    global_angle: float
    position: Dict[str, float]
    endpoint: Dict[str, float]
    limb_length: float
    depth: int


@dataclass
class Connection:
    """Represents a connection between limbs"""
    child_id: int
    parent_id: int
    depth: int
    calibration: Dict[str, float]


class SimulationClient:
    """Client for interacting with the simulator server"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.last_update_time = None
        self.last_data_hash = None
    
    def _get(self, endpoint: str) -> Dict:
        """Make GET request to server"""
        response = requests.get(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get complete simulation state (limbs + connections)"""
        return self._get('/api/simulation_state')
    
    def get_limb_poses(self) -> Dict[str, Any]:
        """Get limb poses with global positions and angles"""
        data = self._get('/api/limb_poses')
        return {
            'limbs': [LimbPose(**limb) for limb in data.get('limbs', [])],
            'count': data.get('count', 0)
        }
    
    def get_connections(self) -> Dict[str, Any]:
        """Get connection tree with calibrations"""
        data = self._get('/api/connections')
        return {
            'connections': [Connection(**conn) for conn in data.get('connections', [])],
            'count': data.get('count', 0)
        }
    
    def get_history(self) -> Dict[str, Any]:
        """Get pose history"""
        return self._get('/api/history')
    
    def get_health(self) -> Dict[str, Any]:
        """Check if server has data"""
        return self._get('/api/health')
    
    def is_connected(self) -> bool:
        """Check if server is running and has data"""
        try:
            health = self.get_health()
            return health.get('status') == 'ok' and health.get('has_data', False)
        except:
            return False
    
    def has_new_data(self) -> bool:
        """Check if simulation has new data since last check"""
        try:
            health = self.get_health()
            current_time = health.get('last_update')
            if current_time != self.last_update_time:
                self.last_update_time = current_time
                return True
            return False
        except:
            return False
    
    def clear_history(self) -> None:
        """Clear pose history"""
        response = requests.post(f"{self.base_url}/api/clear")
        response.raise_for_status()
        print("History cleared")
    
    def get_calibrations(self) -> Dict[int, Dict[str, float]]:
        """Get calibration offsets for each limb"""
        connections = self.get_connections()['connections']
        calibrations = {}
        for conn in connections:
            calibrations[conn.child_id] = conn.calibration
        return calibrations
    
    def get_root_limbs(self) -> List[int]:
        """Find limbs with no parent (depth 0)"""
        connections = self.get_connections()['connections']
        all_children = {conn.child_id for conn in connections}
        all_parents = {conn.parent_id for conn in connections}
        roots = all_parents - all_children
        return list(roots)
    
    def add_noise_to_angles(self, limbs: List[LimbPose], angle_std: float = 2.0) -> List[Dict]:
        """Add Gaussian noise to local angles for GBP observations"""
        import random
        noisy_limbs = []
        for limb in limbs:
            noisy_angle = limb.local_angle + random.gauss(0, angle_std)
            noisy_angle = noisy_angle % 360
            if noisy_angle < 0:
                noisy_angle += 360
            noisy_limbs.append({
                'id': limb.id,
                'local_angle': noisy_angle,
                'depth': limb.depth
            })
        return noisy_limbs
    
    def run_gbp_loop(self, poll_interval: float = 0.1, verbose: bool = True):
        """
        Run GBP loop that executes each time new simulation data arrives.
        
        Args:
            poll_interval: How often to check for new data (seconds)
            verbose: Print status messages
        """
        print("Starting GBP loop...")
        print("Press Ctrl+C to stop")
        print()
        
        iteration = 0
        while True:
            if self.has_new_data():
                iteration += 1
                
                state = self.get_simulation_state()
                limbs = self.get_limb_poses()['limbs']
                connections = self.get_connections()['connections']
                
                if verbose:
                    print(f"Iteration {iteration}: {len(limbs)} limbs, {len(connections)} connections")
                
                # =========================================================
                # TODO: Add your GBP implementation here
                # =========================================================
                # Example:
                # - Get noisy observations: self.add_noise_to_angles(limbs)
                # - Build factor graph from limbs and connections
                # - Run GBP iterations
                # - Extract calibration estimates
                # - Compute error vs ground truth
                # =========================================================
                
                # Placeholder for GBP logic
                self._run_gbp_iteration(limbs, connections, iteration)
            
            time.sleep(poll_interval)
    
    def _run_gbp_iteration(self, limbs: List[LimbPose], connections: List[Connection], iteration: int):
        """Placeholder - replace with your GBP implementation"""
        # Example: compute endpoint positions from FK
        for limb in limbs:
            # This would be replaced with GBP belief computation
            pass
        
        # Print sample data for verification
        if iteration == 1:
            print(f"  Sample limb: id={limbs[0].id}, angle={limbs[0].local_angle:.1f}°, "
                  f"endpoint=({limbs[0].endpoint['x']:.1f}, {limbs[0].endpoint['y']:.1f})")
            if connections:
                print(f"  Sample connection: child={connections[0].child_id}, "
                      f"parent={connections[0].parent_id}, "
                      f"calibration=({connections[0].calibration['offset_x']:.1f}, "
                      f"{connections[0].calibration['offset_y']:.1f})")


client = SimulationClient()


def main():
    print("=" * 60)
    print("Robot Simulator API Client")
    print("=" * 60)
    print()
    
    try:
        health = client.get_health()
        if health.get('status') != 'ok':
            print("Server not responding properly")
            return
        if not health.get('has_data'):
            print("Server running but no simulation data yet.")
            print("Open http://localhost:5000 in browser to run simulation.")
            return
        print(f"Connected to server (last update: {health.get('last_update', 'unknown')})")
    except requests.exceptions.ConnectionError:
        print("Could not connect to server")
        print("Make sure server.py is running: python3 server.py")
        return
    
    state = client.get_simulation_state()
    print(f"  Limbs: {state.get('limb_count', 0)}")
    print(f"  Connections: {state.get('connection_count', 0)}")
    
    print()
    print("-" * 60)
    print("Limb Poses")
    print("-" * 60)
    
    limb_data = client.get_limb_poses()
    for limb in limb_data['limbs']:
        print(f"  Limb {limb.id}: angle={limb.local_angle:.1f}° (global {limb.global_angle:.1f}°)")
        print(f"    Position: ({limb.position['x']:.1f}, {limb.position['y']:.1f})")
        print(f"    Endpoint: ({limb.endpoint['x']:.1f}, {limb.endpoint['y']:.1f})")
    
    print()
    print("-" * 60)
    print("Connections")
    print("-" * 60)
    
    conn_data = client.get_connections()
    for conn in conn_data['connections']:
        print(f"  {conn.child_id} -> {conn.parent_id} (depth {conn.depth}): "
              f"offset=({conn.calibration['offset_x']:.1f}, {conn.calibration['offset_y']:.1f})")
    
    print()
    print("=" * 60)
    print("Usage for GBP Implementation:")
    print("=" * 60)
    print("""
# Option 1: Run the built-in GBP loop
# (edit _run_gbp_iteration to add your GBP logic)
client.run_gbp_loop(poll_interval=0.1)

# Option 2: Manual loop
while True:
    if client.has_new_data():
        limbs = client.get_limb_poses()['limbs']
        connections = client.get_connections()['connections']
        
        # Your GBP code here
        
    time.sleep(0.1)
""")


if __name__ == "__main__":
    main()