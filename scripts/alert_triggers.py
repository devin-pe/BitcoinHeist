import requests
import sys
import time
import random
from multiprocessing import Pool
from configs.configs import ApiConfig


def trigger_http_errors(num_requests: int = 20):
    for i in range(num_requests):
        response = requests.post(
            f"{ApiConfig.api_uri}/predict",
            json={"length": 1.0}, 
            timeout=5
        )
        print(f"Request {i+1}: Status {response.status_code}")
        time.sleep(0.1)


def send_spike_request(i):
    payload = {
        "length": random.randint(1, 20),
        "weight": random.uniform(0.001, 10),
        "count": random.randint(1, 1000),
        "looped": random.randint(0, 10),
        "neighbors": random.randint(1, 100),
        "income": random.uniform(0, 100000)
    }
    response = requests.post(f"{ApiConfig.api_uri}/predict", json=payload, timeout=10)
    return f"Request {i}: {response.status_code}"


def trigger_request_spike(num_requests: int = 150):
    with Pool(processes=10) as pool:
        results = pool.map(send_spike_request, range(num_requests))
        for result in results:
            print(result)


def trigger_data_quality_issues(num_requests: int = 50):
    for i in range(num_requests):
        payload = {
            "length": -1,
            "weight": -1,
            "count": -1,
            "looped": -1,
            "neighbors": -1,
            "income": -1
        }
        
        response = requests.post(f"{ApiConfig.api_uri}/predict", json=payload, timeout=10)
        print(f"Request {i+1}: {response.status_code}")
        time.sleep(0.1)


def trigger_cache_misses(num_requests: int = 30):
    for i in range(num_requests):
        fake_request_id = f"uuid-{i}"
        response = requests.get(f"{ApiConfig.api_uri}/explain/{fake_request_id}", timeout=10)
        print(f"Request {i+1}: Status {response.status_code} - Cache miss")
        time.sleep(0.1)


def trigger_psi_drift(num_requests: int = 120):
    for i in range(num_requests):
        payload = {
            "length": random.uniform(50, 100),
            "weight": random.uniform(20, 50), 
            "count": random.randint(500, 1000),
            "looped": random.randint(5, 10),
            "neighbors": random.randint(50, 100),
            "income": random.uniform(500000, 1000000)
        }
        response = requests.post(f"{ApiConfig.api_uri}/predict", json=payload, timeout=10)
        print(f"Request {i+1}/{num_requests}: Status {response.status_code}")
        if i % 10 == 0:
            time.sleep(0.1)


def run_all_tests():
    tests_dict = {
        "http_errors": trigger_http_errors,
        "request_spike": trigger_request_spike,
        "data_quality": trigger_data_quality_issues,
        "cache_misses": trigger_cache_misses,
        "psi_drift": trigger_psi_drift,
    }
    
    for test_name, test_func in tests_dict.items():
        print(f"Running: {test_name}")
        try:
            test_func()
            print(f"{test_name} completed")
        except Exception as e:
            print(f"{test_name} failed: {e}")
        
        print("Waiting 2 seconds before next test...")
        time.sleep(2)

    print("All tests completed!")


if __name__ == "__main__":
    tests_dict = {
        "http_errors": trigger_http_errors,
        "request_spike": trigger_request_spike,
        "data_quality": trigger_data_quality_issues,
        "cache_misses": trigger_cache_misses,
        "psi_drift": trigger_psi_drift,
        "all": run_all_tests
    }
    
    test_name = sys.argv[1].lower()
    
    if test_name in tests_dict:
        print(f"Running test: {test_name}")
        tests_dict[test_name]()
    else:
        print(f"Unknown test: {test_name}")