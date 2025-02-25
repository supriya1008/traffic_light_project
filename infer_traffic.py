import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import PPO

# Load trained YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load trained RL model
rl_model = PPO.load("models/ppo_traffic.zip")


# Define vehicle and emergency vehicle classes
vehicle_classes = {"car", "bus", "truck", "motorcycle"}
emergency_classes = {"ambulance", "fire truck"}

# Load input traffic video
cap = cv2.VideoCapture("traffic_video.mp4")
output_video = cv2.VideoWriter("output_traffic.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                               (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = yolo_model(frame)

    vehicle_count = 0
    emergency_detected = 0  # 1 if emergency vehicle detected, else 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            if confidence > 0.5:
                label = yolo_model.names[class_id]

                if label in vehicle_classes:
                    vehicle_count += 1

                if label in emergency_classes:
                    emergency_detected = 1  # Emergency vehicle detected

    # Prepare RL input
    obs = np.array([vehicle_count, emergency_detected], dtype=np.float32).reshape(1, -1)

    # Get RL model decision
    action, _ = rl_model.predict(obs)
    action_map = {0: "Reduce Red", 1: "Keep Same", 2: "Increase Red"}
    decision = action_map.get(action[0], "Unknown")

    # Display results on video
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.putText(frame, f"Emergency: {'Yes' if emergency_detected else 'No'}", 
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Signal Decision: {decision}", (30, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    output_video.write(frame)
    cv2.imshow("Traffic Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
