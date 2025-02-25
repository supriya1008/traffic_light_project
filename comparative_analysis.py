import numpy as np
import matplotlib.pyplot as plt
from infer_traffic import predict_signal

# Simulate vehicle traffic patterns
vehicle_counts = np.linspace(10, 100, 10)
emergency_cases = [0, 1]

results = {"Traditional": [], "RL-PPO": []}

for vehicles in vehicle_counts:
    for emergency in emergency_cases:
        # Traditional method (Fixed timing)
        traditional_decision = "Keep Same" if vehicles > 50 else "Reduce Red"
        results["Traditional"].append(traditional_decision)

        # RL-PPO Decision
        rl_decision = predict_signal(int(vehicles), emergency)
        results["RL-PPO"].append(rl_decision)

# Plot comparison
x_labels = [f"{int(v)}{'E' if e else ''}" for v in vehicle_counts for e in emergency_cases]
plt.figure(figsize=(12, 6))
plt.plot(x_labels, results["Traditional"], label="Traditional Control", marker="o")
plt.plot(x_labels, results["RL-PPO"], label="RL-PPO Control", marker="s")
plt.xticks(rotation=45)
plt.ylabel("Signal Decision")
plt.title("Traditional vs RL Traffic Control Decisions")
plt.legend()
plt.show()
