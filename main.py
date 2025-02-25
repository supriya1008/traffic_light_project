from train_ppo import model
from infer_traffic import predict_signal
from comparative_analysis import results

# Example Test
vehicle_count = 30
emergency_detected = 1

decision = predict_signal(vehicle_count, emergency_detected)
print(f"🚦 Final Traffic Decision: {decision}")

# Run Comparative Analysis
print("🔍 Running Comparative Analysis...")
