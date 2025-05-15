import cv2
import numpy as np
import pytesseract

# Configure pytesseract executable path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'

def detect_license_plate_from_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny edge detection
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:
                plate = frame[y:y+h, x:x+w]
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
                plate_text = pytesseract.image_to_string(plate_thresh, config='--psm 8')
                plate_text = plate_text.strip()
                if plate_text:
                    print("Detected License Plate Number:", plate_text)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("License Plate", plate)
                break  # Only process the first detected plate

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_license_plate_from_frame(frame)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
