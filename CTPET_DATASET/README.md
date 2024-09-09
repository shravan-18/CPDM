# A sample of PET-CT Data

## Overview
This folder contains DICOM files for CT and PET scans of 1 study, and pre-processed paired data stored in `.npy` format. The data is organized into folders for raw DICOM files and paired 3D data, with accompanying Jupyter notebooks to demonstrate how to load and process the data.

### Folder Structure
```plaintext
├── Auto-Paired
│   ├── ct.npy
│   ├── pet.npy
├── DICOM
│   ├── [CT and PET DICOM files]
├── read_npy.ipynb
├── read_dicom.ipynb
```

### 1. Folder: `Auto-Paired`

This folder contains the pre-processed paired CT and PET data saved in `.npy` format. The `.npy` files store 3D image data for CT and PET with the following details:

- **`ct.npy`**: Contains the 3D CT image data with shape `(C, H, W)`, where:
  - `C`: Number of slices in the series
  - `H`: Height of 1 slice
  - `W`: Width of 1 slice
- **`pet.npy`**: Contains the 3D PET image data with the same shape `(C, H, W)` as `ct.npy`. The slices in `pet.npy` correspond directly to the slices in `ct.npy` (paired by index `0 .. C-1`).

### 2. Folder: `DICOM`

This folder contains the original CT and PET files in the DICOM format, which have not yet been paired. Each file contains:
- **Metadata** in the DICOM header, including details such as:
  - Patient Sex, Patient Weight
  - Acquisition parameters (e.g., KVP, Rescale Slope, Rescale Intercept)
- **Pixel data**: The actual 2D image slices for each CT and PET scan.

The DICOM files need to be processed individually to extract both metadata and pixel data.

### 3. File: `read_npy.ipynb`

This Jupyter notebook provides example code to load and process the paired `.npy` files from the `Auto-Paired` folder. It demonstrates how to:
- Load `.npy` files into NumPy arrays.
- Visualize individual slices from the 3D data (both CT and PET).

### 4. File: `read_dicom.ipynb`

This Jupyter notebook provides example code to read and process the DICOM files from the `DICOM` folder. It demonstrates how to:
- Load DICOM files using the `pydicom` library.
- Extract metadata (e.g., patient sex, weight) and pixel data from DICOM headers.
- Visualize the 2D slices from CT and PET scans.
