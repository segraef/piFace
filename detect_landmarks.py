import cv2
import dlib
import numpy as np

# Load facial landmark detector and target image
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
target_image = cv2.imread("known_faces/Laura.jpg")

if target_image is None:
    print("Error: Target image not found.")
    exit()

# Function to get facial landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

# Function to apply affine transform to a triangle
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# Function to check if two rectangles intersect
def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

# Function to warp the target face
def warp_face(source_landmarks, target_landmarks, target_image, frame):
    if len(source_landmarks) != len(target_landmarks):
        print("Error: Source and target landmarks count mismatch.")
        return np.zeros_like(frame)

    # Define the Delaunay triangulation
    rect = cv2.boundingRect(target_landmarks)
    subdiv = cv2.Subdiv2D(rect)
    for pt in target_landmarks:
        subdiv.insert((pt[0], pt[1]))

    # Retrieve triangles from Delaunay
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # Initialize the warped image
    warped_image = np.zeros_like(frame)

    for triangle in triangles:
        pts = triangle.reshape(3, 2)
        triangle_rect = cv2.boundingRect(pts)

        if not rectangles_intersect(triangle_rect, rect):
            continue

        # Find the corresponding indices of the triangle vertices
        indices = []
        for pt in pts:
            idx = np.where((target_landmarks == pt).all(axis=1))[0]
            if len(idx) > 0:
                indices.append(idx[0])
        if len(indices) != 3:
            continue

        # Source and destination triangles
        src_tri = source_landmarks[indices].astype(np.int32)
        dst_tri = target_landmarks[indices].astype(np.int32)

        # Bounding rectangles
        src_rect = cv2.boundingRect(np.array([src_tri]))
        dst_rect = cv2.boundingRect(np.array([dst_tri]))

        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            continue

        # Crop and normalize triangles
        src_cropped = frame[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        src_tri_cropped = src_tri - src_rect[:2]
        dst_tri_cropped = dst_tri - dst_rect[:2]

        # Apply affine transformation
        size = (dst_rect[2], dst_rect[3])
        warped_triangle = apply_affine_transform(src_cropped, src_tri_cropped, dst_tri_cropped, size)

        # Create a mask for the triangle
        mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), 255)

        # Overlay the warped triangle on the output image
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask)
        warped_image[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = \
            cv2.add(warped_image[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]], warped_triangle)

    return warped_image

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get landmarks for source face (your face)
    source_landmarks = get_landmarks(frame)
    target_landmarks = get_landmarks(target_image)

    if source_landmarks is not None:
        # Draw landmarks on the webcam feed
        for (x, y) in source_landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    if source_landmarks is not None and target_landmarks is not None:
        # Warp the target face to follow your facial movements
        warped_face = warp_face(source_landmarks, target_landmarks, target_image, frame)

        # Display the warped still image in a separate window
        cv2.imshow("Warped Still Image", warped_face)

        # Overlay warped face on the webcam frame (optional)
        frame = cv2.addWeighted(frame, 0.6, warped_face, 0.4, 0)

    # Display the frame
    cv2.imshow("Facial Reenactment", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
