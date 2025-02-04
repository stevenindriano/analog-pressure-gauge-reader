import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# Constants and Configuration
# -------------------------------
METER_SHAPE = [512, 512]
CIRCLE_CENTER = [256, 256]
CIRCLE_RADIUS = 250
PI = math.pi
RECTANGLE_HEIGHT = 120
RECTANGLE_WIDTH = 1570

METER_CONFIG = {
    'range': 6.0,                       # Range of the meter
    'unit': "Bar"                     # Unit displayed
}

# -------------------------------
# Initialize YOLO Models
# -------------------------------
def initialize_yolo_models(det_model_path, seg_model_path):
    return YOLO(det_model_path), YOLO(seg_model_path)

# -------------------------------
# Upload Image
# -------------------------------
def upload_image(label):
    uploaded_file = st.file_uploader(f"Choose an image for {label}", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = f"temp_image_{label}.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        return temp_path
    return None

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# Crop and Resize Regions (Object Detection)
# -------------------------------
def crop_and_resize_regions(detection_results, image):
    roi_imgs, locations = [], []
    for result in detection_results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            sub_img = image[y_min:y_max, x_min:x_max]
            if sub_img.size > 0:
                resized_img = cv2.resize(sub_img, (METER_SHAPE[0], METER_SHAPE[1]))
                roi_imgs.append(resized_img)
                locations.append([x_min, y_min, x_max, y_max])
    return roi_imgs, locations

# -------------------------------
# Segment Regions (Segmentation)
# -------------------------------
def segment_regions(segmentation_model, regions):
    segmented_results = []
    for region in regions:
        results = segmentation_model.predict(region, conf=0.2)
        for result in results:
            seg_mask = result.plot(labels=False, probs=False, boxes=False, conf=False,
                                   img=np.zeros_like(result.orig_img))
            segmented_results.append(seg_mask)
    return segmented_results

# ------------------------------- 
# Circular to Rectangle Conversion
# -------------------------------
def convert_circle_to_rectangle(segmented_results):
    rectangles = []
    for label_map in segmented_results:
        rectangle = np.zeros((RECTANGLE_HEIGHT, RECTANGLE_WIDTH, 3), dtype=np.uint8)
        for row in range(RECTANGLE_HEIGHT):
            for col in range(RECTANGLE_WIDTH):
                theta = 2 * math.pi * col / RECTANGLE_WIDTH
                rho = CIRCLE_RADIUS - row - 1
                y = int(CIRCLE_CENTER[0] + rho * math.cos(theta))
                x = int(CIRCLE_CENTER[1] - rho * math.sin(theta))
                if 0 <= x < label_map.shape[1] and 0 <= y < label_map.shape[0]:
                    rectangle[row, col] = label_map[y, x]
        rectangles.append(rectangle)
    return rectangles

# -------------------------------
# Erosion Function
# -------------------------------
def erode(seg_results, erode_kernel):
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    eroded_results = []
    for seg_result in seg_results:
        eroded_result = cv2.erode(seg_result.astype(np.uint8), kernel)
        eroded_results.append(eroded_result)
    return eroded_results

# -------------------------------
# Rectangle to Line Conversion
# -------------------------------
def rectangle_to_line(rectangle_meters):
    line_scales, line_pointers = [], []
    for rectangle_meter in rectangle_meters:
        height, width = rectangle_meter.shape[:2]
        line_scale = np.zeros(width, dtype=np.uint8)
        line_pointer = np.zeros(width, dtype=np.uint8)
        for col in range(width):
            for row in range(height):
                r, g, b = rectangle_meter[row, col]
                if r >= 100 and g <= 50:  # Pointer region
                    line_scale[col] += 1
                elif g >= 100:  # Scale region
                    line_pointer[col] += 1
        line_scales.append(line_scale)
        line_pointers.append(line_pointer)
    return line_scales, line_pointers

def mean_binarization(data_list):
    binarized_list = []
    for data in data_list:
        mean_value = np.mean(data)
        thresholded_data = (data > mean_value).astype(np.uint8)
        binarized_list.append(thresholded_data)
    return binarized_list

# -------------------------------
# Locate Scale Positions
# -------------------------------
def locate_scale(line_scales):
    scale_locations = []
    for line_scale in line_scales:
        locations = []
        find_start = False
        one_scale_start = 0
        for j in range(len(line_scale) - 1):
            if line_scale[j] > 0 and line_scale[j + 1] > 0:
                if not find_start:
                    one_scale_start = j
                    find_start = True
            elif find_start and line_scale[j] == 0 and line_scale[j + 1] == 0:
                one_scale_end = j - 1
                locations.append((one_scale_start + one_scale_end) / 2)
                find_start = False
        scale_locations.append(locations)
    return scale_locations

# -------------------------------
# Locate Pointer Positions
# -------------------------------
def locate_pointer(line_pointer):
    if not isinstance(line_pointer, (list, np.ndarray)):
        raise ValueError("Input 'line_pointer' must be a list or a numpy array.")
    
    pointer_start = None
    midpoints = []

    line_pointer = np.ravel(np.asarray(line_pointer))

    for i in range(len(line_pointer)):
        value = line_pointer[i]
        if value > 0 and pointer_start is None:
            pointer_start = i
        elif value == 0 and pointer_start is not None:
            pointer_end = i - 1
            midpoint = (pointer_start + pointer_end) / 2
            midpoints.append(midpoint)
            pointer_start = None
    
    return midpoints

# -------------------------------
# Calculate Reading
# -------------------------------
def get_relative_location(scale_locations, pointer_locations):
    pointed_scales = []
    for scale_location, pointer_location in zip(scale_locations, pointer_locations):
        num_scales = len(scale_location)
        pointed_scale = -1
        if num_scales > 0:
            for i in range(num_scales - 1):
                if scale_location[i] <= pointer_location < scale_location[i + 1]:
                    pointed_scale = i + (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05) + 1
        pointed_scales.append({'num_scales': num_scales, 'pointed_scale': pointed_scale})
    return pointed_scales

def calculate_reading(pointed_scales, min_scale, max_scale):
    scale_range = max_scale - min_scale
    readings = []
    for ps in pointed_scales:
        if ps['pointed_scale'] != -1:
            reading = min_scale + (ps['pointed_scale'] / ps['num_scales']) * scale_range
            readings.append(reading)
        else:
            readings.append(None)  # Handle invalid readings
    return readings

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Pressure Gauge Reading (OPRP)")

    # Load Models
    detection_model, segmentation_model = initialize_yolo_models("meter_det_model.pt", "meter_seg_model.pt")

    # Upload Images
    st.subheader("Upload Images")
    input_image_path = upload_image("Input")
    output_image_path = upload_image("Output")

    if input_image_path and output_image_path:
        try:
            # Input Min and Max Scale Values
            min_scale = st.number_input("Enter the minimum scale value:", value=0.0)
            max_scale = st.number_input("Enter the maximum scale value:", value=6.0)

            # Process Input Image
            st.subheader("Input Image Processing")
            input_image = preprocess_image(input_image_path)
            st.image(input_image, caption="Input Image", use_column_width=True)

            input_detection_results = detection_model.predict(input_image, conf=0.5)
            input_roi_imgs, input_locations = crop_and_resize_regions(input_detection_results, input_image)
            input_segmented_results = segment_regions(segmentation_model, input_roi_imgs)
            input_eroded_results = erode(input_segmented_results, erode_kernel=4)
            input_rectangles = convert_circle_to_rectangle(input_eroded_results)
            input_line_scales, input_line_pointers = rectangle_to_line(input_rectangles)
            input_scale_locations = locate_scale(mean_binarization(input_line_scales))
            input_pointer_locations = locate_pointer(mean_binarization(input_line_pointers))
            input_pointed_scales = get_relative_location(input_scale_locations, input_pointer_locations)
            input_readings = calculate_reading(input_pointed_scales, min_scale, max_scale)

            # Process Output Image
            st.subheader("Output Image Processing")
            output_image = preprocess_image(output_image_path)
            st.image(output_image, caption="Output Image", use_column_width=True)

            output_detection_results = detection_model.predict(output_image, conf=0.5)
            output_roi_imgs, output_locations = crop_and_resize_regions(output_detection_results, output_image)
            output_segmented_results = segment_regions(segmentation_model, output_roi_imgs)
            output_eroded_results = erode(output_segmented_results, erode_kernel=4)
            output_rectangles = convert_circle_to_rectangle(output_eroded_results)
            output_line_scales, output_line_pointers = rectangle_to_line(output_rectangles)
            output_scale_locations = locate_scale(mean_binarization(output_line_scales))
            output_pointer_locations = locate_pointer(mean_binarization(output_line_pointers))
            output_pointed_scales = get_relative_location(output_scale_locations, output_pointer_locations)
            output_readings = calculate_reading(output_pointed_scales, min_scale, max_scale)

            # Display Results
            st.subheader("Results")
            for i, (input_reading, output_reading) in enumerate(zip(input_readings, output_readings)):
                if input_reading is not None and output_reading is not None:
                    st.write("**Pressure Input:**")
                    st.write(f"{input_reading:.2f}")
                    st.write("**Pressure Output:**")
                    st.write(f"{output_reading:.2f}")
                else:
                    st.warning(f"Meter {i+1}: Invalid Reading")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()