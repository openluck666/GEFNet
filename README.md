# GIFNet 

-------------------------------------------------
##### **Frameworks**

![show](https://github.com/openluck666/GEFNet/blob/main/framework.png?raw=true)

-------------------------------------------------
##### **Results**

*multi-exposure fusion*
![show](https://github.com/openluck666/GEFNet/blob/main/experience.png?raw=true)



##### **Our dataset**

The dataset including training can be download from [SICE](https://github.com/csjcai/SICE)

Moreover, we also provide results of our method and the comparison methods on [RealHDV](https://github.com/yungsyu99/Real-HDRV) 、 [MEFB](https://github.com/xingchenzhang/MEFB)、 [UltraFusion](https://github.com/OpenImagingLab/UltraFusion) and [MSEC](https://github.com/mahmoudnafifi/Exposure_Correction)

-------------------------------------------------
##### **We provide a simple training and testing process as follows:**

-------------------------------------------------
##### **Dependencies**

* Python 3.8

* PyTorch 1.10.0+cu113

  You can setup the required Anaconda environment by running the following prompts:

  ```cpp
  conda create -n GEFNet python=3.8.17
  conda activate GEFNet
  pip install -r requirements.txt
  ```

  

-------------------------------------------------
##### **Train**

The datasets samples are placed in *images\dataset* (including MEFB[1], MFIF[2], VIFB[3], and SICE[4]).

> Multi-Exposure Image Fusion (MEF)

```python
python train.py --config 1
```

Then, the checkpoints and log file are saved in *output*.

-------------------------------------------------
##### **Test**

The pretrained models are placed in *ckp*.

> MEF

```python
python test.py --config 1 --ckp mef.pth
```



-------------------------------------------------
Finally, the fused results can be found in images\fused*.

-------------------------------------------------
