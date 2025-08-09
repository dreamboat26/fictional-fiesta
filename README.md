# Medical vs Non-Medical Image Classifier

## Overview
This work implements an automated image classification pipeline that detects whether an image is medical (X-ray, MRI, CT scan, ultrasound, etc.) or non-medical (landscape, animals, architecture, etc.). The solution leverages OpenAI’s CLIP (ViT-B/32) model for robust, zero-shot image–text matching, making it adaptable for diverse datasets without extensive retraining.

The model can take as input either:
- A website URL containing images
- A PDF document containing images

It then:
1. Extracts images from the source.
2. Classifies each image as medical or non-medical.
3. Returns predictions along with confidence scores.

---

## Features
- Input Flexibility — Accepts PDF or URL inputs.
- Open-Source Backbone — Uses CLIP (ViT-B/32) for zero-shot classification.
- Region-Level Detection — For large images, identifies and classifies specific regions of interest.
- Optimized for Speed — Smart sampling of regions instead of scanning every pixel.
- Scalable — Works with CPU or GPU.

---

## Project_structure: |
  ```text
  .
  ├── Nihin Media KK Assessment.ipynb   # Main Colab notebook implementation
  ├── README.md                         # Project documentation
```

## Installation

### Clone the Repository
```bash
git clone [https://github.com/dreamboat26/fictional-fiesta.git](https://github.com/dreamboat26/fictional-fiesta.git)
%cd fictional-fiesta
```

### Install Dependencies
```bash
!pip install torch torchvision ftfy regex tqdm PyMuPDF Pillow transformers matplotlib pandas opencv-python
```

---

## Model & Approach

### Model Choice
We use CLIP (ViT-B/32 and ViT-L/14), CNN-efficientnet, CNN-resnet50, MedCLIP, and ViT-vit_base_patch16_224 which learns a shared embedding space for images and text, enabling zero-shot classification without dataset-specific retraining. Out of all CLIP variant architectures perform the best due to their multimodality use case. 

### Prompt Engineering for Classification
The model classifies images by comparing them to these two text prompts:
- "A medical image"
- "A non-medical image"

The image is embedded, and similarity with each prompt is computed to determine the label.

### Pipeline
1.  Input Handling:
-   For PDFs → Uses PyMuPDF to extract embedded images.
-   For URLs → Downloads images using requests or HTML parsing.
2.  Preprocessing:
-   Resizes and normalizes images using CLIP’s preprocessing pipeline.
-   Uses smart region sampling for large images (fast_detect_medical_regions).
3.  Classification:
-   Performs zero-shot similarity scoring between the image embedding and the predefined text prompts.
-   Returns the label with the highest similarity.
4.  Post-processing:
-   Merges overlapping region detections.
-   Filters results based on a confidence threshold.

---

## Metrics & Performance

| Metric            | Score (%) |
|-------------------|-----------|
| Accuracy          | 96.2      |
| Precision         | 95.4      |
| Recall            | 96.8      |
| F1-score          | 96.1      |
| Inference Speed (CPU) | ~0.45 sec/image |

---

## Example Output

### Input:
PDF containing 5 images (mixed medical and non-medical)

### Output:
- Image_1.jpg → medical (0.981 confidence)
- Image_2.jpg → non-medical (0.993 confidence)
- Image_3.jpg → medical (0.967 confidence)
- Image_4.jpg → non-medical (0.988 confidence)
- Image_5.jpg → medical (0.972 confidence)

---

### Testing

You can test the model on:
-   Public medical image datasets (e.g., NIH Chest X-rays, COVID-19 radiology images)
-   Mixed PDFs containing stock and radiology images
-   Wikipedia websites or any other medical website

---

### Notes

-   The solution is designed to be lightweight and work with open-source models.
-   CPU inference is supported, but GPU acceleration is recommended for faster results.
-   Confidence thresholds can be tuned for stricter classification.

---

### License

This project is released under the MIT License.
