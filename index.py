import cv2
import numpy as np

# Function to apply Canny Edge Detection
def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blur, 50, 150)  # Apply Canny edge detector
    return edges

# Function to define a region of interest (mask)
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]  # Define the polygon region
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)  # Fill the polygon with white
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to display yellow lines for both left and right lanes
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                if all(isinstance(coord, int) for coord in [x1, y1, x2, y2]):  # Ensure all coordinates are integers
                    # Ensure yellow color (BGR format) is applied
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 12)  # Yellow color for both left and right lines
    return line_image

# Function to average the slope and intercept of lines
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None  # If no lines detected, return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a 1st degree polynomial (line)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:  # Negative slope = left lane
            left_fit.append((slope, intercept))
        else:  # Positive slope = right lane
            right_fit.append((slope, intercept))

    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None

    # Return only valid lines (exclude None)
    return [line for line in [left_line, right_line] if line is not None]

# Function to make coordinates for lines
def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None

    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except ZeroDivisionError:
        return None

# Main function to process video frame-by-frame
def process_frame(frame):
    edges = canny_edge_detection(frame)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    
    # Check if valid lines are found before drawing them
    if averaged_lines is not None and len(averaged_lines) > 0:
        line_image = display_lines(frame, averaged_lines)
        combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # Blend line image with original frame
    else:
        combined_image = frame
    
    return combined_image

# Capture video from a file or use '0' for webcam
#video_path = 'C:/Users/Ram/Downloads/solidYellowLeft.mp4'  # Replace with your video path or use 0 for webcam
video_path = 'C:\\Users\\Ram\\Downloads\\video.mp4'
 
cap = cv2.VideoCapture(video_path)

# Ensure video is opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))  # Resize to a manageable size
    lane_frame = process_frame(frame)
    
    cv2.imshow('Lane Line Detection', lane_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
