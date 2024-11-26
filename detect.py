import cv2
import face_recognition
import os

# Load known face encodings and their names
known_face_encodings = []
known_face_names = []

# Path to the folder containing known face images
KNOWN_FACES_DIR = "known_faces"

# Load and encode known faces if the folder is not empty
if not os.listdir(KNOWN_FACES_DIR):
    print("No known faces found in the 'known_faces' directory. Running without recognition.")
else:
    for file_name in os.listdir(KNOWN_FACES_DIR):
        image_path = os.path.join(KNOWN_FACES_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file_name)[0])  # Use the file name as the person's name
        name = os.path.splitext(file_name)[0]
        print(f"Loaded and encoded face: {name}")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # If we have known faces, find matches
    face_encodings = []
    if known_face_encodings:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings or []):
        name = "Unknown"

        # Check if the face matches any known faces
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

        # Draw rectangle and label around the face
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4  # Scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Print details in terminal
        print(f"Detected: {name}, Location: X:{left}, Y:{top}, Width:{right-left}, Height:{bottom-top}")

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
