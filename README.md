# LiCGNet: Lightweight Context Guided Network for Polyp Segmentation on Edge Devices

## Introduction 

This research presents a lightweight and deployable AI model for real-time polyp segmentation, designed specifically for edge devices in resource-limited clinical settings. By enabling fast, and accurate segmentation, it supports early diagnosis of colorectal cancer which is highly generizable on multiple polyp datasets (considered 5 datasets in this experiment), a crucial factor in improving patient survival rates. Leveraging the Context-Guided Network (CGNet), the model achieves 98.86% accuracy and 86.02% mIoU on the CVC-300 dataset. It is optimized for deployment on devices like Raspberry Pi 5, and supports FP16 quantization, drastically reducing model size by half with negligible accuracy loss. Achieving up to 94 FPS on GPU, this work marks a step forward in democratizing AI for real-time, point-of-care diagnostics.

## Installation
1) Clone this Repository
```
https://github.com/kaditya123/Polyp_Segmentation_LiCGNet.git
cd Polyp_Segmentation_LiCGNet
```
2) Install the required packages through below command
```
pip install -r requirements.txt
```
3) Dataset is already splitted within ```dataset/Polyp_Processed_1_0_data``` folder, and the ttesting splits can be found in ```dataset/Polyp_Processed_1_0_data/TESTING_Image+Mask/```. Download the datasets from original sources as per below.
   - [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
   - [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
   - [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579?file=37636550)
   - [CVC-300](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579?file=37636550)
   - [ETIS-Larib](https://link.springer.com/article/10.1007/s11548-013-0926-3)

*⚠️ This model is intended for research and educational purposes only and it is not directly recommended for clinical deployments until various approvals and validations.*
