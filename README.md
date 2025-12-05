# [ICCV 2025] Controllable-LPMoE: Adapting to Challenging Object Segmentation via Dynamic Local Priors from Mixture-of-Experts

Yanguang Sun, Jiawei Lian, Jian Yang, Lei Luo<br />

Our work has been accepted for **ICCV 2025**. The relevant code and results will be updated gradually.

If you are interested in our work, please do not hesitate to contact us at **Sunyg@njust.edu.cn via email**.

![image](https://github.com/user-attachments/assets/9871e846-712e-406d-b6d8-c6338284eb89)
<img width="2028" height="675" alt="image" src="https://github.com/user-attachments/assets/ccbad38e-a1e9-49a9-97f0-46e6b033a60d" />
<img width="1991" height="782" alt="image" src="https://github.com/user-attachments/assets/2408651d-04ab-4a37-b2be-b53c5f3d149b" />
<img width="1996" height="505" alt="image" src="https://github.com/user-attachments/assets/856312f0-07e9-4fb4-b161-7ce0024fb4e3" />
<img width="1997" height="843" alt="image" src="https://github.com/user-attachments/assets/74f25990-8571-4235-9470-3051df2f67a1" />
<img width="1979" height="969" alt="image" src="https://github.com/user-attachments/assets/01ad3f53-7f7a-450e-823f-f40c09bf0d48" />

# Segmentation results

We provide the predicted results genereted by our Controllable-LPMoE model across six **binary object segmentation** tasks, including **''Camouflaged Object Detection (COD)''**, **''Salient Object Detection (SOD)''**, **''Polyp Segmentation (PS)''**, **''Skin Lesion Segmentation (SLS)''**, **''Shadow Detection (SD)''**, and **''Glass Detection (GD)''**.

LPMoE_U_B_ICCV25_COD [(https://pan.baidu.com/s/1KABwnsRhw75Wecya1RQ6Fw?pwd=bicb), PIN:bicb] 

LPMoE_U_B_ICCV25_SOD [(https://pan.baidu.com/s/18Hf5KTqyZliLgv30qGEJvw?pwd=ysst), PIN:ysst] 

LPMoE_U_B_ICCV25_PS [(https://pan.baidu.com/s/1gwPD7ti9OnpGuyOIgB_oeQ), PIN:2nff] 

LPMoE_U_B_ICCV25_SLS [(https://pan.baidu.com/s/1LZuOXTFBHmo6ka3RhzfRTg), PIN:zpbh] 

LPMoE_U_B_ICCV25_SD [(https://pan.baidu.com/s/1kpTEFNSSYqBCW7bRwEpApQ), PIN:csim] 

LPMoE_U_B_ICCV25_GD [(https://pan.baidu.com/s/1CcD3AAKMSTiYo3HHWoFtrg), PIN:yxqc] 


## Training

To train Controllable_LPMoE on COD on a single node with 4 gpus run:

```shell

bash dist_train.sh configs/COS/Controllable_LPMoE_COD_Beit.py 4 --seed 2024

```

## Image Demo

The segmentation results can be obtained through image_demo.


# Citation

If you use Controllable-LPMoE in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```
@inproceedings{Controllable-LPMoE,
  title={Controllable-LPMoE: Adapting to Challenging Object Segmentation via Dynamic Local Priors from Mixture-of-Experts},
  author={Sun, Yanguang and Lian, Jiawei and Yang, Jian and Luo, Lei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22327--22337},
  year={2025}
}
```

















