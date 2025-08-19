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
3) Dataset is already splitted within ```dataset/Polyp_Processed_1_0_data``` folder, and the testing splits can be found in ```dataset/Polyp_Processed_1_0_data/TESTING_Image+Mask/```. Splitting criteria is mentioned as per the table, and, datasets can be download from original sources from links below.
<table align="center" border="1" cellspacing="0" cellpadding="8" style="border-collapse:collapse; text-align:center; width:100%;">
  <tr>
    <th>Dataset</th>
    <th>Number</th>
    <th>Train + Val</th>
    <th>Test</th>
    <th>Resolution</th>
  </tr>
  <tr>
    <td align="left"><b> <a href="https://datasets.simula.no/kvasir-seg/" title="Kvasir-SEG Dataset">Kvasir-SEG</a> </b> </td>
    <td>1000</td>
    <td>900</td>
    <td>100</td>
    <td>1070 × 1348</td>
  </tr>
  <tr>
    <td align="left"><b> <a href="https://polyp.grand-challenge.org/CVCClinicDB/" title="CVC-ClinicDB Dataset">CVC-ClinicDB</a> </b> </td>
    <td>612</td>
    <td>550</td>
    <td>62</td>
    <td>288 × 384</td>
  </tr>
  <tr>
    <td align="left"><b>  <a href="https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579?file=37636550" title="CVC-ColonDB Dataset">CVC-ColonDB</a> </b></td>
    <td>380</td>
    <td>-</td>
    <td>380</td>
    <td>500 × 574</td>
  </tr>
  <tr>
    <td align="left"><b> <a href="https://link.springer.com/article/10.1007/s11548-013-0926-3" title="ETIS-Larib Dataset">ETIS-Larib</a> </b></td>
    <td>196</td>
    <td>-</td>
    <td>196</td>
    <td>966 × 1225</td>
  </tr>
  <tr>
    <td align="left"><b> <a href="https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579?file=37636550" title="CVC-300 Dataset">CVC-300</a></b></td>
    <td>60</td>
    <td>-</td>
    <td>60</td>
    <td>500 × 574</td>
  </tr>
</table>



## Model Training
1) Training on the 'Train' set
```
python Kvasir_CVCclinic_train.py --dataset KvasirSeg_CVCClinic_(train) --train_type ontrain --train_data_list ./dataset/Polyp_processed_1_0_data/Polyp_train_list.txt --max_epochs 250
```
2) Training on the 'Train + Val' set
```
python Kvasir_CVCclinic_train.py --dataset KvasirSeg_CVCClinic_(train+val) --train_type ontrainval --train_data_list ./dataset/Polyp_processed_1_0_data/Polyp_train&val_list.txt --max_epochs 250
```
## Model Inferencing
1) Testing the model on *Kvasir-SEG* testing set
```
python Kvasir_CVCclinic_test.py --test_data_list ./dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_Kvasir_SEG_test_list.txt --resume ./checkpoint/KvasirSeg_CVCClinic_(train)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth
```
Set ```--resume ./checkpoint/KvasirSeg_CVCClinic_(train+val)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth``` if training is done 'train + val' dataset.

2) Testing the model on *CVC-ClinicDB* testing set
```
python Kvasir_CVCclinic_test.py --test_data_list ./dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_CVC_ClinicDB_test_list.txt --resume ./checkpoint/KvasirSeg_CVCClinic_(train)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth
```

3) Testing the model on *CVC-ColonDB* testing set
```
python Kvasir_CVCclinic_test.py --test_data_list ./dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_CVC_ColonDB_test_list.txt --resume ./checkpoint/KvasirSeg_CVCClinic_(train)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth
```

4) Testing the model on *CVC-300* testing set
```
python Kvasir_CVCclinic_test.py --test_data_list ./dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_CVC_300_test_list.txt --resume ./checkpoint/KvasirSeg_CVCClinic_(train)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth
```

5) Testing the model on *ETIS-Larib* testing set
```
python Kvasir_CVCclinic_test.py --test_data_list ./dataset/Polyp_processed_1_0_data/TESTING_Image+Mask/Polyp_ETIS_LaribPolypDB_test_list.txt --resume ./checkpoint/KvasirSeg_CVCClinic_(train)/CGNet_M3N21bs8gpu1_ontrain/model_250.pth
``` 


*⚠️ This work is not directly recommended for clinical deployments until significant approvals and validations.*
