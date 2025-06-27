# IDRiD Dataset Structure

This dataset is organized for two main tasks:
- **A. Segmentation**: For lesion and optic disc segmentation.
- **B. Disease Grading**: For diabetic retinopathy and risk of macular edema grading.

All data is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

---

## A. Segmentation

### 1. Original Images

- **Path**: `dataset/A. Segmentation/1. Original Images/`
- **Subfolders**:
  - `a. Training Set/`: 54 color fundus images (`IDRiD_01.jpg` to `IDRiD_54.jpg`)
  - `b. Testing Set/`: 27 color fundus images (`IDRiD_55.jpg` to `IDRiD_81.jpg`)
- **Format**: JPEG images

### 2. All Segmentation Groundtruths

- **Path**: `dataset/A. Segmentation/2. All Segmentation Groundtruths/`
- **Subfolders**:
  - `a. Training Set/`
  - `b. Testing Set/`
- **Each set contains 5 folders (one per class):**
  1. `1. Microaneurysms/` (`*_MA.tif`)
  2. `2. Haemorrhages/` (`*_HE.tif`)
  3. `3. Hard Exudates/` (`*_EX.tif`)
  4. `4. Soft Exudates/` (`*_SE.tif`)
  5. `5. Optic Disc/` (`*_OD.tif`)
- **Format**: TIFF binary masks, one per image per class. File names match the original images.

---

## B. Disease Grading

### 1. Original Images

- **Path**: `dataset/B. Disease Grading/1. Original Images/`
- **Subfolders**:
  - `a. Training Set/`: 413 images (`IDRiD_001.jpg` to `IDRiD_413.jpg`)
  - `b. Testing Set/`: 103 images (`IDRiD_001.jpg` to `IDRiD_103.jpg`)
- **Format**: JPEG images

### 2. Groundtruths

- **Path**: `dataset/B. Disease Grading/2. Groundtruths/`
- **Files**:
  - `a. IDRiD_Disease Grading_Training Labels.csv`
  - `b. IDRiD_Disease Grading_Testing Labels.csv`
- **Format**: CSV files with the following columns:
  - `Image name`: e.g., `IDRiD_001`
  - `Retinopathy grade`: Integer (0–4), where higher is more severe
  - `Risk of macular edema`: Integer (0–2), where higher is more severe

---

## Licensing

- **Authors**: Prasanna Porwal, Samiksha Pachade, and Manesh Kokare
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **See**: `LICENSE.txt` and `CC-BY-4.0.txt` in each main folder for details.

---

## Example Structure

```plaintext
dataset/
  ├── A. Segmentation/
  │   ├── 1. Original Images/
  │   │   ├── a. Training Set/
  │   │   └── b. Testing Set/
  │   └── 2. All Segmentation Groundtruths/
  │       ├── a. Training Set/
  │       │   ├── 1. Microaneurysms/
  │       │   ├── 2. Haemorrhages/
  │       │   ├── 3. Hard Exudates/
  │       │   ├── 4. Soft Exudates/
  │       │   └── 5. Optic Disc/
  │       └── b. Testing Set/
  │           └── (same as above)
  └── B. Disease Grading/
      ├── 1. Original Images/
      │   ├── a. Training Set/
      │   └── b. Testing Set/
      └── 2. Groundtruths/
          ├── a. IDRiD_Disease Grading_Training Labels.csv
          └── b. IDRiD_Disease Grading_Testing Labels.csv
```

---

## Notes

- All images are color fundus photographs.
- Segmentation masks are binary TIFFs, one per lesion type per image.
- Disease grading labels are provided as CSVs, with integer grades for each image.
- The dataset is suitable for both segmentation and classification tasks in diabetic retinopathy research. 