import cv2
import argparse
import numpy as np # Added for potential math operations if needed later

def highlightFace(net, frame, conf_threshold=0.7):
    """
    Detects faces in an image and draws bounding boxes around them.
    """
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # Draw a green rectangle around the face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2) # thickness=2
    return frameOpencvDnn, faceBoxes

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Age and Gender Detection from Image or Webcam")
    parser.add_argument('--image', type=str, help="Path to the input image file. If not provided, webcam will be used.")
    parser.add_argument('--face_conf', type=float, default=0.7, help="Minimum confidence threshold for face detection (0.0 to 1.0).")
    args = parser.parse_args()

    # Define paths to the model files
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # Mean values for preprocessing, specific to the Caffe models used
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Predefined lists for age and gender categories
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-27)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load the pre-trained neural networks
    try:
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
    except cv2.error as e:
        print(f"Error loading one or more network models: {e}")
        print("Please ensure the model files are in the correct path and are not corrupted.")
        return

    # Open video capture: webcam or image file
    # If args.image is provided, use it; otherwise, use webcam (0)
    video_source = args.image if args.image else 0
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    # Padding around the detected face for better classification
    # Using relative padding: 20% of the face width/height
    padding_factor = 0.2 
    min_face_size = 30 # Minimum dimension for a face ROI to be processed

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            if args.image: # If it's an image, we've processed it
                print("Finished processing image.")
            else: # Webcam stream might have ended or encountered an error
                print("No frame received from webcam or video stream ended.")
            cv2.waitKey(0) # Wait indefinitely if it's an image or error
            break

        # Detect faces in the current frame
        resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold=args.face_conf)
        if not faceBoxes:
            if not args.image: # Only print for webcam if continuously no face
                 print("No face detected in the current frame")
        else:
            print(f"Detected {len(faceBoxes)} face(s)")

        # Process each detected face
        for faceBox in faceBoxes:
            # Extract the face region of interest (ROI)
            face_x1, face_y1, face_x2, face_y2 = faceBox
            
            # Calculate padding based on face size
            face_width = face_x2 - face_x1
            face_height = face_y2 - face_y1
            current_padding_x = int(padding_factor * face_width)
            current_padding_y = int(padding_factor * face_height)

            # Get the region for the face, applying padding and ensuring it's within frame bounds
            roi_x1 = max(0, face_x1 - current_padding_x)
            roi_y1 = max(0, face_y1 - current_padding_y)
            roi_x2 = min(frame.shape[1] - 1, face_x2 + current_padding_x)
            roi_y2 = min(frame.shape[0] - 1, face_y2 + current_padding_y)
            
            face = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # Check if the extracted face ROI is valid
            if face.size == 0 or face.shape[0] < min_face_size or face.shape[1] < min_face_size:
                print(f"Skipping faceBox {faceBox} due to invalid or too small ROI: shape {face.shape}")
                continue
            
            # --- Preprocessing for Age and Gender Networks ---
            # Create a blob from the face ROI
            # Input size (227,227) is specific to these age/gender Caffe models
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # To use grayscale (some models might prefer it, but these Caffe models usually expect BGR):
            # gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # gray_face_resized = cv2.resize(gray_face, (227, 227))
            # blob = cv2.dnn.blobFromImage(gray_face_resized, 1.0, (227,227), (0,0,0), swapRB=False, crop=False) 
            # Note: Mean values might need to be adjusted for grayscale, often (0,0,0) or a single mean value

            # --- Gender Prediction ---
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender_index = genderPreds[0].argmax()
            gender = genderList[gender_index]
            gender_confidence = genderPreds[0].max() # Get confidence of the top prediction

            # --- Age Prediction ---
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age_index = agePreds[0].argmax()
            age = ageList[age_index]
            age_confidence = agePreds[0].max() # Get confidence of the top prediction

            # Display the predicted gender and age on the image
            # label = f"{gender} ({gender_confidence*100:.1f}%), {age} ({age_confidence*100:.1f}%)"
            label = f"{gender}, {age}" # Simpler label
            
            # Position the label slightly above the face bounding box
            label_y_pos = faceBox[1] - 10 if faceBox[1] - 10 > 10 else faceBox[1] + 10
            cv2.putText(resultImg, label, (faceBox[0], label_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("Age and Gender Detection", resultImg)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # ESC or 'q' key to exit
            break
        if args.image and key != -1: # If it's an image, any key press (other than special) can exit
            break


    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()