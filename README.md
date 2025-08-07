# LiCGNet: Lightweight Context Guided Network for Polyp Segmentation on Edge Devices

## Introduction 

This research presents a lightweight and deployable AI model for real-time polyp segmentation, designed specifically for edge devices in resource-limited clinical settings. By enabling fast, and accurate segmentation, it supports early diagnosis of colorectal cancer which is highly generizable on multiple polyp datasets (considered 5 datasets in this experiment), a crucial factor in improving patient survival rates. Leveraging the Context-Guided Network (CGNet), the model achieves 98.86% accuracy and 86.02% mIoU on the CVC-300 dataset. It is optimized for deployment on devices like Raspberry Pi 5, and supports FP16 quantization, drastically reducing model size by half with negligible accuracy loss. Achieving up to 94 FPS on GPU, this work marks a step forward in democratizing AI for real-time, point-of-care diagnostics.

*⚠️ This model is intended for research and educational purposes only and it is not directly recommended for clinical deployments until various approvals and validations.*
