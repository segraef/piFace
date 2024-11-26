import cv2
import dlib
import numpy as np

# Load facial landmark detector and target image
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
target_image = cv2.imread("known_faces/Seb.jpg")

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
    return cv2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# Function to check if two rectangles intersect
def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def warp_face(source_landmarks, target_landmarks, source_image, target_image):
    if len(source_landmarks) != len(target_landmarks):
        print("Error: Source and target landmarks count mismatch.")
        return target_image

    # Create a copy of the target image to modify
    output_image = target_image.copy()

    # Create a mask for the warped face
    mask = np.zeros_like(target_image, dtype=np.uint8)

    # Warp each triangle from the source to the target
    rect = cv2.boundingRect(np.array(target_landmarks, dtype=np.int32))
    subdiv = cv2.Subdiv2D(rect)
    for pt in target_landmarks:
        subdiv.insert((pt[0], pt[1]))

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for triangle in triangles:
        pts = triangle.reshape(3, 2)

        # Skip invalid triangles
        if any((pt[0] < 0 or pt[1] < 0 or pt[0] >= target_image.shape[1] or pt[1] >= target_image.shape[0]) for pt in pts):
            continue

        indices = []
        for pt in pts:
            idx = np.where((np.array(target_landmarks) == pt).all(axis=1))[0]
            if len(idx) > 0:
                indices.append(idx[0])
        if len(indices) != 3:
            continue

        src_tri = np.array([source_landmarks[i] for i in indices], dtype=np.float32)
        dst_tri = np.array([target_landmarks[i] for i in indices], dtype=np.float32)

        # Calculate affine transform
        M = cv2.getAffineTransform(src_tri, dst_tri)
        size = (rect[2], rect[3])
        warped_triangle = cv2.warpAffine(source_image, M, size)

        # Mask the warped triangle
        triangle_mask = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, np.int32(dst_tri - rect[:2]), (255, 255, 255))

        # Add the warped triangle to the mask
        mask[rect[1]:rect[1] + size[1], rect[0]:rect[0] + size[0]] = cv2.add(
            mask[rect[1]:rect[1] + size[1], rect[0]:rect[0] + size[0]],
            cv2.bitwise_and(warped_triangle, triangle_mask),
        )

    # Blend the warped face mask with the target image
    output_image = cv2.addWeighted(output_image, 1, mask, 0.7, 0)

    return output_image


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
        for x, y in source_landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Warp the target face based on the source landmarks and target landmarks
    if source_landmarks is not None and target_landmarks is not None:
        # Create a copy of the target image to display in the second window
        warped_target_image = warp_face(source_landmarks, target_landmarks, frame, target_image)

        # Display the full target image with the warped face
        cv2.imshow("Warped Target Image", warped_target_image)

    # Display the webcam feed with landmarks (optional)
    if source_landmarks is not None:
        for (x, y) in source_landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow("Facial Reenactment", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
