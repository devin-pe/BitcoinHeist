# White Call
curl \
-H "Content-Type: application/json" \
-d '{"length":1.0, "weight":4.0, "count":2.0, "looped":0.0, "neighbors":2.0, "income":100.0}' \
 -X POST http://127.0.0.1:5001/predict

# Ransomware Call
curl \
-H "Content-Type: application/json" \
-d '{"length":18, "weight":0.00833333333333333, "count":1, "looped":0, "neighbors":2, "income":100050000}' \
 -X POST http://127.0.0.1:5001/predict

# Negative Datapoint Call
curl \
-H "Content-Type: application/json" \
-d '{"length":1.0, "weight":4.0, "count":2.0, "looped":0.0, "neighbors":2.0, "income":-100.0}' \
 -X POST http://127.0.0.1:5001/predict