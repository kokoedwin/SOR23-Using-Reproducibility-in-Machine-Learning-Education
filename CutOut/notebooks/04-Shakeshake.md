
::: {.cell .markdown}
### ShakeShake
:::

::: {.cell .markdown}

Shake-shake regularization model implementation from https://github.com/xgastaldi/shake-shake

Note: for faster training, use Runtime > Change Runtime Type to run this notebook on a GPU.
:::


::: {.cell .markdown}

In the Cutout paper, the authors claim that:

1. Cutout improves the robustness and overall performance of convolutional neural networks.
2. Cutout can be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.

In this section, we will evaluate these claims using a WideResNet model. For the WideResNet model, the specific quantitative claims are given in the following table: