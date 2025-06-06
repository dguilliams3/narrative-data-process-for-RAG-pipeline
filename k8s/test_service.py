import requests
import time
from datetime import datetime

def test_health():
    """Test the health endpoint of the GPT service."""
    try:
        response = requests.get('http://localhost:30000/health')
        print(f"[{datetime.now()}] Health check status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"[{datetime.now()}] Health check failed: {str(e)}")
        return False

def main():
    """Run health checks every 5 seconds."""
    print("Starting health checks...")
    while True:
        test_health()
        time.sleep(5)

if __name__ == "__main__":
    main() 