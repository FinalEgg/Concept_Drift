# run.py
from __init__ import create_app
from simulation_device.sim_device import SimDevice
import threading

app = create_app()

def start_sim_device():
    sim_device = SimDevice(host='127.0.0.1', port=9000)
    sim_device.start()

if __name__ == '__main__':
    sim_thread = threading.Thread(target=start_sim_device, daemon=True)
    sim_thread.start()
    app.run(debug=True)