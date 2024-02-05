# Sparse-VQ Transformer

An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting

 Our Sparse-VQ capitalizes
on a sparse vector quantization technique coupled with Reverse
Instance Normalization (RevIN) to reduce noise impact and capture sufficient statistics for forecasting, serving as an alternative
to the Feed-Forward layer (FFN) in the transformer architecture.
Our FFN-free approach trims the parameter count, enhancing computational efficiency and reducing overfitting. Through evaluations across ten benchmark datasets, including the newly introduced CAISO dataset, Sparse-VQ surpasses leading models with
a 7.84% and 4.17% decrease in MAE for univariate and multivariate time series forecasting, respectively. 




## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain the datasets from [[Autoformer](https://github.com/thuml/Autoformer)] .
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:


```bash
bash ./scripts/traffic_M.sh
bash ./scripts/traffic_S.sh
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/MAZiqing/FEDformer

https://github.com/lucidrains/vector-quantize-pytorch
