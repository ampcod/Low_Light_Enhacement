import tkinter as tk
from tkinter import filedialog, messagebox
import zipfile
import os
import numpy as np
import cv2
import rasterio
import xml.etree.ElementTree as ET
from rasterio.transform import from_bounds
import threading
import customtkinter as ctk

# Initialize global variables
bounding_box = None
extracted_folder = None

def unzip_folder(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        messagebox.showinfo("Unzip Completed", "Files unzipped successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while unzipping: {e}")

def parse_metadata(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        namespaces = {
            'pds': 'http://pds.nasa.gov/pds4/pds/v1',
            'disp': 'http://pds.nasa.gov/pds4/disp/v1',
            'isda': 'https://isda.issdc.gov.in/pds4/isda/v1',
            'sp': 'http://pds.nasa.gov/pds4/sp/v1',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        def get_text(element, default=None):
            return element.text if element is not None else default

        metadata = {
            'projection': None,
            'upper_left': None,
            'upper_right': None,
            'lower_left': None,
            'lower_right': None,
            'width': None,
            'height': None,
            'data_type': None
        }

        projection = root.find('.//isda:projection', namespaces=namespaces)
        metadata['projection'] = get_text(projection)

        metadata['upper_left'] = (
            float(get_text(root.find('.//isda:upper_left_latitude', namespaces=namespaces), 0.0)),
            float(get_text(root.find('.//isda:upper_left_longitude', namespaces=namespaces), 0.0))
        )
        metadata['upper_right'] = (
            float(get_text(root.find('.//isda:upper_right_latitude', namespaces=namespaces), 0.0)),
            float(get_text(root.find('.//isda:upper_right_longitude', namespaces=namespaces), 0.0))
        )
        metadata['lower_left'] = (
            float(get_text(root.find('.//isda:lower_left_latitude', namespaces=namespaces), 0.0)),
            float(get_text(root.find('.//isda:lower_left_longitude', namespaces=namespaces), 0.0))
        )
        metadata['lower_right'] = (
            float(get_text(root.find('.//isda:lower_right_latitude', namespaces=namespaces), 0.0)),
            float(get_text(root.find('.//isda:lower_right_longitude', namespaces=namespaces), 0.0))
        )

        width = int(get_text(root.find('.//pds:Axis_Array[pds:axis_name="Sample"]/pds:elements', namespaces=namespaces)))
        height = int(get_text(root.find('.//pds:Axis_Array[pds:axis_name="Line"]/pds:elements', namespaces=namespaces)))
        metadata['width'] = width
        metadata['height'] = height

        data_type = get_text(root.find('.//pds:Element_Array/pds:data_type', namespaces=namespaces))
        metadata['data_type'] = data_type

        return metadata
    except ET.ParseError:
        messagebox.showerror("XML Parsing Error", "Error parsing XML metadata.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while parsing metadata: {e}")

def read_disk_image(image_file, width, height, data_type):
    try:
        dtype = np.uint8 if data_type == 'UnsignedByte' else np.float32
        data = np.fromfile(image_file, dtype=dtype)
        data = data.reshape((height, width))
        return data
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading the image file: {e}")

def create_geotiff(data, metadata, output_file):
    try:
        upper_left = metadata['upper_left']
        upper_right = metadata['upper_right']
        lower_left = metadata['lower_left']
        lower_right = metadata['lower_right']

        left = upper_left[1]
        right = upper_right[1]
        top = upper_left[0]
        bottom = lower_left[0]

        transform = from_bounds(left, bottom, right, top, data.shape[1], data.shape[0])
        crs = 'EPSG:4326'

        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while creating GeoTIFF: {e}")

def show_loader(root, message):
    loader = tk.Toplevel(root)
    loader.title("Processing")
    loader.geometry("300x100")
    label = tk.Label(loader, text=message)
    label.pack(pady=20)
    loader.transient(root)
    loader.grab_set()
    return loader

def browse_zip():
    global extracted_folder
    zip_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
    if zip_path:
        folder_path = filedialog.askdirectory(title="Select folder to extract ZIP")
        if folder_path:
            unzip_folder(zip_path, folder_path)
            tiff_button.config(state=tk.NORMAL)
            enhance_button.config(state=tk.NORMAL)
            extracted_folder = folder_path

def process_tiff():
    folder_path = filedialog.askdirectory(title="Select folder containing XML and IMG files")
    
    if folder_path:
        xml_file = None
        img_file = None
        
        for file in os.listdir(folder_path):
            if file.endswith(".xml"):
                xml_file = os.path.join(folder_path, file)
            elif file.endswith(".img"):
                img_file = os.path.join(folder_path, file)
        
        if xml_file and img_file:
            tiff_output = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])
            
            if tiff_output:
                loader = show_loader(root, "Creating TIFF file, please wait...")

                def create_tiff():
                    try:
                        metadata = parse_metadata(xml_file)
                        if metadata:
                            data = read_disk_image(img_file, metadata['width'], metadata['height'], metadata['data_type'])
                            create_geotiff(data, metadata, tiff_output)
                            messagebox.showinfo("TIFF Created", f"GeoTIFF file created: {tiff_output}")
                    finally:
                        loader.destroy()

                threading.Thread(target=create_tiff).start()

        else:
            messagebox.showerror("File Not Found", "Could not find XML or IMG files in the selected folder.")

def single_scale_retinex(img, sigma):
    """Apply Single Scale Retinex to the image."""
    try:
        retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
        return retinex
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while applying Retinex: {e}")

def apply_clahe(image, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)

def enhance_crater_area(image, bounding_box=None, clip_limit=3.0):
    # Initialize default bounding box values
    x_min, y_min, x_max, y_max = 0, 0, image.shape[1], image.shape[0]

    # Use the provided bounding box if available
    if bounding_box:
        x_min, y_min, x_max, y_max = bounding_box

        # Validate bounding box dimensions
        if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
            raise ValueError("Bounding box exceeds image dimensions")
    else:
        # If no bounding box is provided, return the original image
        return image

    # Create a copy of the original image to ensure non-bounding box areas remain unchanged
    result_image = image.copy()

    # Crop the image using the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Apply Single Scale Retinex
    smoothed_image = single_scale_retinex(cropped_image, sigma=50)

    # Normalize the result to 8-bit grayscale
    smoothed_image = cv2.normalize(smoothed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Noise Reduction
    smoothed_image = cv2.fastNlMeansDenoising(smoothed_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply CLAHE for contrast enhancement
    smoothed_image = apply_clahe(smoothed_image, clip_limit)

    # Ensure shapes match
    if smoothed_image.shape == (y_max - y_min, x_max - x_min):
        # Replace the bounding box area in the result image with the enhanced portion
        result_image[y_min:y_max, x_min:x_max] = smoothed_image
    else:
        raise ValueError(f"Shape mismatch: smoothed_image is {smoothed_image.shape}, "
                         f"but the target slice in result_image is {(y_max - y_min, x_max - x_min)}")

    return result_image

def enhance_image(image_path, clip_limit):
    """Enhance the entire image with CLAHE."""
    try:
        with rasterio.open(image_path) as dataset:
            profile = dataset.profile
            profile.update(dtype=rasterio.uint8, count=1)

            # Read the entire image at once
            image = dataset.read(1).astype(np.float32)

            # Apply Single Scale Retinex to the entire image
            retinex_image = single_scale_retinex(image, sigma=100)  # Adjust sigma for better small crater detection

            # Normalize the Retinex result to 8-bit grayscale
            retinex_normalized = cv2.normalize(retinex_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply Noise Reduction to reduce artifacts
            denoised_image = cv2.fastNlMeansDenoising(retinex_normalized, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # Apply Low-Light Enhancement using CLAHE (with stronger parameters)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(denoised_image)

            # Reconstruct crater floor using morphological operations (optional)
            kernel = np.ones((5, 5), np.uint8)
            smoothed_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_CLOSE, kernel)

            # Save the enhanced image as a new GeoTIFF
            output_path = os.path.splitext(image_path)[0] + '_enhanced.tif'
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(smoothed_image, 1)

        messagebox.showinfo("Enhancement Completed", "Image enhancement completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while enhancing the image: {e}")

def process_enhance():
    image_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
    if image_path:
        try:
            clip_limit = float(clip_limit_entry.get())
            loader = show_loader(root, "Enhancement process underway...")
            threading.Thread(target=lambda: enhance_image(image_path, clip_limit)).start()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value for CLAHE clip limit.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing enhancement: {e}")

def process_enhance_Bounding_box():
    image_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
    if image_path:
        try:
            # Get the bounding box values from the user inputs
            x_min = int(x_min_entry.get())
            y_min = int(y_min_entry.get())
            x_max = int(x_max_entry.get())
            y_max = int(y_max_entry.get())

            # Ensure valid bounding box coordinates
            bounding_box = (x_min, y_min, x_max, y_max)

            # Show loader while processing
            loader = show_loader(root, "Enhancement process underway...")

            def enhance_image_with_bbox():
                try:
                    # Read the image using rasterio
                    with rasterio.open(image_path) as dataset:
                        image = dataset.read(1)  # Read the first band of the image
                        profile = dataset.profile

                    # Enhance the specific bounding box area
                    enhanced_image = enhance_crater_area(image, bounding_box=bounding_box, clip_limit=float(clip_limit_entry.get()))

                    # Save the enhanced image back to file
                    output_path = os.path.splitext(image_path)[0] + '_enhanced_bounding_box.tif'
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(enhanced_image, 1)  # Write the enhanced image

                    # Inform the user of completion
                    messagebox.showinfo("Enhancement Completed", f"Enhanced image saved as {output_path}")
                finally:
                    loader.destroy()

            # Run the enhancement process in a separate thread
            threading.Thread(target=enhance_image_with_bbox).start()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for bounding box coordinates.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Setting up the CustomTkinter window
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()  # Initialize the CustomTkinter root window
root.title("Lunar Image Processing Tool")
root.geometry("700x600")

# Left Frame: Information
left_frame = ctk.CTkFrame(root, width=200, height=600, corner_radius=10)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

software_info_label = ctk.CTkLabel(left_frame, text="Lunar Image Processing Tool\n\nThis tool helps in preprocessing and enhancement of lunar images.", anchor="w", justify="left")
software_info_label.pack(padx=10, pady=10)

# Right Upper Frame: Preprocessing
right_upper_frame = ctk.CTkFrame(root, width=800, height=300, corner_radius=10)
right_upper_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

preprocessing_label = ctk.CTkLabel(right_upper_frame, text="Preprocessing Actions", font=("Arial", 16))
preprocessing_label.pack(pady=10)

# Buttons for preprocessing actions
zip_button = ctk.CTkButton(right_upper_frame, text="Select ZIP Folder", command=browse_zip)
zip_button.pack(pady=10)

tiff_button = ctk.CTkButton(right_upper_frame, text="Create TIFF File", command=process_tiff)
tiff_button.pack(pady=10)

# Right Lower Frame: Image Enhancement
right_lower_frame = ctk.CTkFrame(root, width=800, height=300, corner_radius=10)
right_lower_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

enhancement_label = ctk.CTkLabel(right_lower_frame, text="Enhancement Options", font=("Arial", 16))
enhancement_label.pack(pady=10)

# Clip Limit Input for CLAHE
clip_limit_label = ctk.CTkLabel(right_lower_frame, text="Set CLAHE Clip Limit:")
clip_limit_label.pack(pady=5)

clip_limit_entry = ctk.CTkEntry(right_lower_frame)
clip_limit_entry.insert(0, "2.0")
clip_limit_entry.pack(pady=5)

# Full Image Enhancement Button
enhance_button = ctk.CTkButton(right_lower_frame, text="Enhance Full Image", command=process_enhance)
enhance_button.pack(pady=10)

x_min_entry = ctk.CTkEntry(right_lower_frame, placeholder_text="x_min")
x_min_entry.pack(pady=5)

y_min_entry = ctk.CTkEntry(right_lower_frame, placeholder_text="y_min")
y_min_entry.pack(pady=5)

x_max_entry = ctk.CTkEntry(right_lower_frame, placeholder_text="x_max")
x_max_entry.pack(pady=5)

y_max_entry = ctk.CTkEntry(right_lower_frame, placeholder_text="y_max")
y_max_entry.pack(pady=5)

draw_bbox_button = ctk.CTkButton(right_lower_frame, text="Enhance Bounding Box Area", command=process_enhance_Bounding_box)
draw_bbox_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()


