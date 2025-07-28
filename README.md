# GIFNet 

-------------------------------------------------
##### **Frameworks**

![show](https://github.com/openluck666/GEFNet/blob/main/framework.png?raw=true)

-------------------------------------------------
##### **Results**

*multi-exposure fusion*
![show](https://github.com/openluck666/GEFNet/blob/main/experience.png?raw=true)



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
