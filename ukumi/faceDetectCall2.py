import face_recognition
import os
import csv
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

# Create new directory for annotated images and CSV
new_dir = "./txt/annotated_faces_with_emotion"
os.makedirs(new_dir, exist_ok=True)
csv_file = "./txt/face_detection_emotion_results.csv"

_FACE_CSV_HEADER = [
    "image", "top", "right", "bottom", "left", "confidence", "total_faces", "emotion",
    "Facex1","Facey1","Facex2","Facey2","Facex3","Facey3","Facex4","Facey4","fullPath"  # TL -> TR -> BR -> BL
]

# Initialize CSV with headers
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow(["Image_Name", "Top", "Right", "Bottom", "Left", "Confidence", "Total_Faces", "Emotion"])
    writer.writerow(_FACE_CSV_HEADER)


def detect_faces_and_emotions1(image_path):
    """
    Detect faces in a thumbnail image, estimate confidence, recognize emotions,
    draw bounding boxes with confidence and emotion, save annotated image, and log to CSV.
    
    Args:
        image_path (str): Path to the thumbnail image (e.g., JPG/PNG).
    
    Returns:
        list: List of tuples (top, right, bottom, left, confidence, emotion) for each detected face.
              If no face detected or image invalid, returns empty list.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return []
    
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        print(f"Loaded image: {image_path}")
        
        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([os.path.basename(image_path)] + [None]*6 + [0, "None"])  # No faces
            return []
        
        # Detect facial landmarks for confidence estimation
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        
        # Prepare results
        results = []
        total_faces = len(face_locations)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Load a larger font
        font = ImageFont.load_default()  # Default font, can be replaced with a larger one like ImageFont.truetype('arial.ttf', 20)
        try:
            font = ImageFont.truetype("./src/arial.ttf", 20)  # Larger font size, adjust path if needed
        except:
            print("Could not load arial.ttf, using default font")
        
        for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
            # Confidence heuristic: percentage of detected landmark points (68 total points expected)
            landmark_count = sum(len(points) for points in landmarks.values())
            confidence = min(landmark_count / 68.0, 1.0)  # Normalize to 0-1
            
            # Crop face region as NumPy array
            face_img = image[top:bottom, left:right]
            
            # Detect emotion using DeepFace with NumPy array
            try:
                emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = emotion_result[0]['dominant_emotion'] if emotion_result else "Unknown"
            except Exception as e:
                print(f"Emotion detection failed for {image_path}: {e}")
                emotion = "Unknown"
            
            results.append((top, right, bottom, left, confidence, emotion))
            
            # Draw bounding box (green) and text (confidence + emotion)
            draw.rectangle([(left, top), (right, bottom)], outline="green", width=2)
            text = f"{confidence:.2f}\n{emotion}"
            draw.text((left, top - 30), text, fill="green", font=font)
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(new_dir, f"annotated_{base_name}.jpg")
        pil_image.save(output_path, "JPEG")
        
        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for result in results:
                writer.writerow([base_name] + list(result[:5]) + [total_faces] + [result[5]])
        
        return results
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

import os, csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import face_recognition
from deepface import DeepFace

# expects globals:
#   csv_file: path to CSV
#   new_dir:  output folder for annotated images


def _ensure_face_csv(csv_path: str):
    """Create parent folder if needed and ensure header exists."""
    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    if (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(_FACE_CSV_HEADER)

def detect_faces_and_emotions(image_path):
    """
    Detect faces in a thumbnail image, estimate confidence, recognize emotions,
    draw bounding boxes with confidence and emotion, save annotated image, and log to CSV.
    Returns:
        list of tuples: (top, right, bottom, left, confidence, emotion) per detected face.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return []

    _ensure_face_csv(csv_file)

    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        #print(f"Loaded image: {image_path}")

        # Detect face locations
        face_locations = face_recognition.face_locations(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if not face_locations:
            # no faces: write a row with blanks for coordinates
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [base_name] + [None, None, None, None, 0.0, 0, "None"] +  # top,right,bottom,left,conf,total,emotion
                    ["","","","","","","",""]                                  # x1..y4 blanks
                )
            return []

        # Landmarks for confidence heuristic
        face_landmarks = face_recognition.face_landmarks(image, face_locations)

        results = []
        total_faces = len(face_locations)

        # Prepare drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("./src/arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            print("Could not load arial.ttf, using default font")

        # Process each face
        rows_to_write = []
        for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
            # Confidence: fraction of 68 expected points that were returned
            landmark_count = sum(len(points) for points in landmarks.values())
            confidence = float(min(landmark_count / 68.0, 1.0))

            # Crop face region for emotion
            face_img = image[top:bottom, left:right]
            try:
                emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = emotion_result[0]['dominant_emotion'] if emotion_result else "Unknown"
            except Exception as e:
                print(f"Emotion detection failed for {image_path}: {e}")
                emotion = "Unknown"

            # Rectangle corners (clockwise from top-left) to match your other CSV:
            # top = y, right = x, bottom = y, left = x
            x1, y1 = float(left),  float(top)    # TL
            x2, y2 = float(right), float(top)    # TR
            x3, y3 = float(right), float(bottom) # BR
            x4, y4 = float(left),  float(bottom) # BL

            # Collect for CSV
            rows_to_write.append([
                base_name, int(top), int(right), int(bottom), int(left),
                round(confidence, 4), total_faces, emotion,
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                f"{x3:.2f}", f"{y3:.2f}", f"{x4:.2f}", f"{y4:.2f}",
                f"{os.path.abspath(image_path)}"
            ])

            # Draw annotations
            draw.rectangle([(left, top), (right, bottom)], outline="green", width=2)
            text = f"{confidence:.2f} | {emotion}"
            draw.text((left, max(0, top - 22)), text, fill="green", font=font)

            results.append((top, right, bottom, left, confidence, emotion))

        # Save annotated image
        os.makedirs(new_dir, exist_ok=True)
        output_path = os.path.join(new_dir, f"annotated_{base_name}.jpg")
        pil_image.save(output_path, "JPEG")

        # Write all detections for this image to CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

        return results

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []



# Example usage


if __name__ == "__main__":
  
    #thumbnail_path = "/cluster/datastore/aniketag/urumi/metadata/downloads/GregSalazar/shortest_top_150_videos/thumbnails/"

    allFolders = os.listdir("/cluster/datastore/aniketag/urumi/metadata/downloads/")
    
    
    for folder in allFolders:

        thumbnail_path = f"/cluster/datastore/aniketag/urumi/metadata/downloads/{folder}/top_150_videos/thumbnails/"

        if not os.path.isdir(thumbnail_path):
            continue
        for path in os.listdir(thumbnail_path):
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(thumbnail_path, path)
                faces = detect_faces_and_emotions(full_path)
                print(f"Image: {path}, Detected faces with confidence and emotion: {faces}")