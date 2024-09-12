# Enhancement of Permanently Shadowed Regions (PSR) of Lunar Craters Captured by OHRC of Chandrayaan-2
This project enhances low-light images of permanently shadowed regions (PSR) of lunar craters captured by the Orbiter High-Resolution Camera (OHRC) of Chandrayaan-2. The aim is to improve the signal-to-noise ratio (SNR) for better interpretation and analysis of these challenging datasets.

## Overview
Lunar Image Processing Tool is a desktop software enables users to efficiently process and analyze ISRO Chandrayaan-2 OHRC images. It includes features to automate image enhancements, conversion, and allow partial or full image processing with advanced algorithms. The primary goal is to improve the visibility and clarity of lunar crater images, specifically in the dimly lit PSR areas.

### Key Features
**1. Seamless Input Handling-**
Users can directly input OHRC zip files downloaded from the CHMapBrowser.
The software automatically extracts and processes data for analysis.<br/>
**2. Automated GeoTIFF Conversion-**
Effortlessly converts lunar surface image data into high-resolution GeoTIFF format.
Includes spatial metadata for precise geospatial analysis, facilitating integration with GIS tools.<br/>
**3. Full Image Enhancement-**
Enhances the entire lunar surface using advanced image processing algorithms.
Improves overall image quality and clarity for better interpretation of lunar features.<br/>
**4. Partial Image Enhancement-**
Users can select specific regions of interest using bounding boxes.
Only the selected areas are enhanced, allowing for focused analysis while leaving the rest of the image untouched.<br/>
**5. Advanced Image Processing Techniques-**
The software employs the following algorithms for enhancement:
Single Scale Retinex (SSR),Contrast Limited Adaptive Histogram Equalization (CLAHE),Gaussian Blur,Denoising

## How to Use
* Download the OHRC image zip file from *https://chmapbrowse.issdc.gov.in/MapBrowse/*.
* Use the desktop software to input the zip files.
* Select either full image enhancement or partial enhancement (using bounding boxes).
* Run the enhancement process.
* The enhanced image will be saved as a GeoTIFF for further analysis.

## Future Enhancements
*Adding more image processing techniques for better edge detection and crater analysis.
*Integrating machine learning models for automated crater and boulder detection.
*Improve User Interface

[Enhancement Screenshots]![image](https://github.com/user-attachments/assets/555e6369-b39a-497c-bcb5-f86f7d1d1627)
[User Interface]![image](https://github.com/user-attachments/assets/5f5c58b5-43ee-4414-9409-743b11b3bb79)
