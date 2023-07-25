
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

In this section, we will evaluate these claims using ShakeShake regularization. For ShakeShake, the specific quantitative claims are given in the following table:

Test error (%, flip/translation augmentation, mean/std normalization, mean of 3 runs) and “+” indicates standard data augmentation (mirror
+ crop)

| **Network** | **CIFAR-10+** | **CIFAR-100+** |
| ----------- | ------------ | ------------- |
| Shake-shake | 2.86         | 15.58         |
| Shake-shake + cutout | 2.56 | 15.20 |

In the given table, the test errors of the Shake-shake regularization model implemented from a GitHub source on CIFAR-10+ and CIFAR-100+ datasets are presented. The "+" indicates the use of standard data augmentation techniques like mirror and crop.

On the CIFAR-10+ dataset, the use of the Shake-shake regularization model results in a test error of 2.86%. With the application of the cutout augmentation, this error decreases slightly to 2.56%. A similar pattern is noticed with the CIFAR-100+ dataset, where the Shake-shake model results in a test error of 15.58% which gets reduced to 15.20% when cutout augmentation is applied.

These results again underline the effectiveness of cutout augmentation in improving model performance, as indicated by the reduced test error rates. The Shake-shake model with both standard and cutout augmentation techniques performs exceptionally well, particularly on the CIFAR-10+ dataset.

:::