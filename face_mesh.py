import cv2
import mediapipe as mp

# Initialize mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Initialize Face Mesh object
face = mp_face_mesh.FaceMesh()
draw_spaces = mp_draw.DrawingSpec((0,255,0),thickness=2, circle_radius=2)

# point_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green small points
# line_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)   # Red
# Open webcam
cap = cv2.VideoCapture(r"C:\Users\Awais Shakeel\Downloads\3761460-uhd_3840_2160_25fps.mp4")

while cap.isOpened():
    r, frame = cap.read()

    if r:
        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe works in RGB)
        newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face landmarks
        result = face.process(newimg)

        # If landmarks are found
        if result.multi_face_landmarks:
            for landmark in result.multi_face_landmarks:
                # Draw face landmarks and connections
                mp_draw.draw_landmarks(frame,
                                        landmark,
                                        mp_face_mesh.FACEMESH_TESSELATION,
                                        # landmark_drawing_spec=point_spec,
                                        # connection_drawing_spec=line_spec
                                        draw_spaces,draw_spaces
                                          )

        # Show the frame with landmarks
        frame = cv2.resize(frame , (1000,600))
        cv2.imshow('cap', frame)

        # Press 'p' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    else:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
