import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'C:\Users\supriya janjirala\Desktop\AGRI_RASPBERRY\Algae_Detection\best.pt')

def detect_algae(frame):
    results = model(frame)
    return results

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run detection
        results = detect_algae(frame)

        # Process results
        for result in results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Coordinates of the bounding box
                score = box.conf.item()  # Confidence score
                class_id = box.cls.item()  # Class ID
                if score > 0.5:  # Confidence threshold
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Algae: {score:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Algae Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()