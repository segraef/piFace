import cv2
import dlib
import numpy as np

# Load facial landmark detector and target image
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
target_image = cv2.imread("known_faces/Seb.jpg")

# Function to get facial landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()])

# Function to apply affine transform to a triangle
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Function to warp triangles
def warp_face(source_landmarks, target_landmarks, target_image, frame):
    # Ensure landmarks are valid and correctly formatted
    if target_landmarks is None:
        print("No landmarks found in the target image.")
        return np.zeros_like(frame)

    target_landmarks = np.array(target_landmarks, dtype=np.float32)
    source_landmarks = np.array(source_landmarks, dtype=np.float32)

    # Define the Delaunay triangulation
    rect = cv2.boundingRect(target_landmarks)
    subdiv = cv2.Subdiv2D(rect)
    for pt in target_landmarks:
        subdiv.insert((float(pt[0]), float(pt[1])))

    # Retrieve triangles from Delaunay
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # Warp each triangle
    warped_image = np.zeros_like(frame)
    for triangle in triangles:
        pts = triangle.reshape(3, 2)
        if cv2.boundingRect(pts) in rect:
            idx = [np.where((target_landmarks == pt).all(axis=1))[0][0] for pt in pts]
            src_tri = source_landmarks[idx].astype(np.int32)
            dst_tri = target_landmarks[idx].astype(np.int32)

            # Warp each triangle
            src_rect = cv2.boundingRect(np.array([src_tri]))
            dst_rect = cv2.boundingRect(np.array([dst_tri]))
            src_cropped = target_image[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]

            # Normalize coordinates to the cropped region
            src_tri_cropped = src_tri - src_rect[:2]
            dst_tri_cropped = dst_tri - dst_rect[:2]

            # Apply affine transform to the triangle
            size = (dst_rect[2], dst_rect[3])
            warped_triangle = apply_affine_transform(src_cropped, src_tri_cropped, dst_tri_cropped, size)

            # Mask for the triangle
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

    if source_landmarks is not None and target_landmarks is not None:
        # Warp the target face to follow your facial movements
        warped_face = warp_face(source_landmarks, target_landmarks, target_image, frame)

        # Overlay warped face on the original frame
        frame = cv2.addWeighted(frame, 0.6, warped_face, 0.4, 0)

    # Display the frame
    cv2.imshow("Facial Reenactment", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
