2022-10-19
THis project is to complete the segmentation with implicit neural on signl image

origian img:

![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/imgs/rgb.png)


segmentation map:

![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/imgs/color.png)

missing:

![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/imgs/row_mask_color.png)

Our results:

random init + relu activation function:
![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/noinit%2Brelu%2B4900.png) ![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/noinit%2Brelu%2Bloss_all.png)

sin init + relu activation function:
![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/sininit%2Brelu%2B4900.png) ![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/sininit%2Brelu%2Bloss_all.png)

sin init + sin activation function:
![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/sininit%2Bsinactivaction_4900.png) ![Image text](https://github.com/qizipeng/Segmentation_completion/blob/master/results/sininit%2Bsinactivaction%2Bloss_all.png)

The sin init + sin activation functinon takes eight hundreds of training results to pause and resume training to get better results.