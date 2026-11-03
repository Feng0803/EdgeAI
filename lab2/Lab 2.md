---
title: Lab 2

---


# Large Language Model Compression
# Part 1: Quantization (30%)

Quantization maps floating-point values to low-bit integers, which reduces model size and memory traffic and allows integer-only inference on edge devices.
In this lab you will implement the building blocks of **linear quantization** step by step, from quantizing a single tensor to quantized Linear / Conv2d inference, and then explore a few techniques that are used by modern quantization toolchains.

## Objective

* Understand linear quantization $r = S(q - Z)$, and how the scale $S$ and zero point $Z$ are chosen.
* Implement per-channel weight quantization, bias quantization, and quantized `Linear` / `Conv2d` inference.
* Implement group-wise weight quantization, MSE-based clipping, BatchNorm folding, and fake quantization with the Straight-Through Estimator (STE).

## Environment

* We use **PyTorch 2.6.0**. Please install the matching version so that your results agree with the expected outputs of the tests.
* A GPU runtime is required (the notebook moves MobileNetV2 to CUDA). Kaggle (GPU T4) or Google Colab (T4 GPU) both work.

## Questions

| Question | Topic |
|:---:|---|
| Q1 | Linear quantization of a tensor `linear_quantize()` |
| Q2 | Scale and zero point from $[r_{\min}, r_{\max}]$ |
| Q3 | Symmetric scale for weights / per-channel quantization |
| Q4 | Bias quantization ($Z_{\mathrm{bias}}=0$, $S_{\mathrm{bias}} = S_{\mathrm{input}} S_{\mathrm{weight}}$) |
| Q5 | Quantized fully-connected layer `quantized_linear()` |
| Q6 | Quantized convolution layer `quantized_conv2d()` |
| Q7 | Group-wise weight quantization |
| Q8 | Choosing the clipping range (MSE search) |
| Q9 | Folding BatchNorm into convolution |
| Q10 | Fake quantization with Straight-Through Estimator (STE) |


* Only write code between `YOUR CODE STARTS HERE` and `YOUR CODE ENDS HERE`.
* **Do not modify** any function marked **(DO NOT MODIFY!!)**, i.e. the test functions and the given helpers.
* Every question has a test function. A test prints `* Test passed.` when your implementation is correct, then draws a figure that helps you see what your function does
* The MobileNetV2 will be used in this part. You can download it here: [mobilenetv2_0.963.pth](https://drive.google.com/file/d/14Qu5NMqyVAhf6vFxXqx9ysPtfV4gbwM_/view?usp=drive_link)

:::info
Q7 – Q10 reuse `get_quantized_range()`, `linear_quantize()` (Q1), `get_quantization_scale_for_weight()` and `linear_quantize_weight_per_channel()` (Q3). Make sure all tests of Q1 – Q6 pass before you start Q7.
:::


# Environment
To ensure a consistent grading environment, **Part 2 and Part 3** will be evaluated using the following base environment:

```text
Python 3.11
torch==2.5.1+cu121
```

Your implementation must be compatible with this environment.

You may install additional packages as needed, but they must be clearly specified in your `requirements.txt` or setup instructions.

---
# Part 2: LLM compression (50%)
## Objective
In this part, you will compress the **Linear layers of Qwen3-4B**.

All experiments in Part 2 must use the following model:

- Model: `Qwen/Qwen3-4B`
- Hugging Face: https://huggingface.co/Qwen/Qwen3-4B

Your goal is to design a compression strategy that satisfies the given **Compression Ratio** and **Perplexity (PPL)** requirements.

You are free to choose and combine different model compression techniques. The choice of method, compression configuration, and layer-wise strategy are all part of this assignment.

All four cases must start from the same original Qwen3-4B model and must be evaluated using the provided evaluation procedure.

:::danger
**Restrictions**
The following techniques are not allowed as part of your compression strategy:
- KV Cache compression or quantization
- Speculative Decoding
- Parameter Offloading
- Fine-tuning
- Multi-GPU Model Parallelism
- Any technique that reduces memory usage without compressing the model itself
:::

### Compression Ratio
Compression Ratio measures how much the model size is reduced after compression.

It is defined as:

$$\text{Compression Ratio}=1-\frac{\text{Compressed Size}}{\text{Original Size}}$$

A higher Compression Ratio indicates a greater reduction in model size.

For example, a Compression Ratio of **75%** means that the compressed representation requires only **25% of the original storage size**.

In this lab, the Compression Ratio will be calculated based on the target Linear layers. **Any additional parameters introduced by your compression method must also be included in the compressed size.**

### Perplexity(PPL)

Perplexity (PPL) is used to evaluate the language modeling capability of the compressed model.

$$\text{PPL}=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i|x_{<i})\right)$$

A lower PPL indicates better model performance.

Compression usually reduces model size at the cost of some model quality. Therefore, your goal is to achieve the required **Compression Ratio** while keeping the **PPL** below the target value.

All PPL results in this lab must be obtained using the provided `run_eval.py`.

## Compression Tasks
Complete the following four cases.
You may use a different compression strategy for each case.

- Case 1
**Compression Ratio > 75%**
**PPL < 15**

- Case 2
**Compression Ratio >= 81%**
**PPL < 20**

- Case 3
**SVD Compression Ratio >= 20%
Overall Compression Ratio = 50%
PPL < 50**

- Case 4
**Pruning Rate >= 20%
Overall Compression Ratio >= 85%
PPL < 50**


## Submission & Report Requirements

For each case, you are required to design and apply a compression strategy that satisfies the given constraints.

### Evaluation

We will provide a `run_eval` file for evaluating the model.

:::danger
You must use the provided `run_eval` for all PPL evaluations.

- Do **not** modify the PPL evaluation procedure.
- Do **not** replace it with your own evaluation implementation.
- The result produced by the provided evaluation code will be used as the official result.
:::

### Code Submission

Your submitted code must include at least the following files:

```text
YourID_Part2/
├── requirements.txt
├── run.sh
├── run_eval.py
└── ...
```

### `requirements.txt` and `run.sh`

The `requirements.txt` file should contain all Python packages required to reproduce your results.

The provided `run.sh` must serve as the **main entry point** of your submission. It should automatically:

1. Install all required packages from `requirements.txt`.
2. Run the compression procedure for all four cases.
3. Evaluate all four cases using the provided `run_eval.py`.
4. Print the final Compression Ratio and PPL for each case.

Your submission should be executable using:

```bash
bash run.sh
```

No additional commands should be required before running `run.sh`.

The final output should contain only the **Compression Ratio** and **PPL** of each case.

For example:

```text
Case 1
Compression Ratio: 76.3%
PPL: 14.21

Case 2
Compression Ratio: 82.1%
PPL: 18.57

Case 3
Compression Ratio: 50.4%
PPL: 31.82

Case 4
Compression Ratio: 86.0%
PPL: 42.15
```

## Report

The report should explain how you completed each of the four cases.

For **each case**, include:

1. **Compression Strategy**
   - What compression method(s) you used
   - Which Linear layers were compressed
   - Important settings or hyperparameters

2. **Result**
   - Final Compression Ratio
   - Final PPL

3. **Analysis**
    - Explain the relationship between your compression strategy and the obtained result
   

At the end of the report, provide an overall conclusion based on the four cases.

You may discuss:

- The trade-off between Compression Ratio and PPL
- Which Linear layers are more sensitive to compression
- Differences between the compression strategies used in different cases
- Why certain configurations perform better than others
- The limitations of your approach

Finally, summarize what you learned from the experiments and which compression strategy you consider the most effective under the given constraints.

---
# Part 3: Calibration-Free Quantization Reproduction(20%)

## Objective

In this part, you will implement a **calibration-free Post-Training Quantization (PTQ)** method and reproduce its model-quality result.

The following paper is provided as the reference method:

**[AdaptQuant: Hardware-Aligned Per-Tile MX Format Selection for Post-Training Quantization on NVIDIA Blackwell](https://openreview.net/forum?id=ZGeR4gkEZq)**

You may reproduce AdaptQuant or choose another PTQ method that satisfies the requirements of this assignment.

The main focus of this part is **target-model calibration-free quantization**. The quantization method must not require a calibration dataset or target-model-specific calibration procedure to determine quantization parameters, thresholds, precision allocation, or other compression configurations.

Since the experiments will be conducted on NVIDIA T4 GPUs, hardware-specific low-precision kernels are **not required**. You may use **fake quantization** to simulate the numerical behavior of the target low-precision format and evaluate the resulting model quality.

The performance score will be determined by how closely your reproduced result matches the corresponding result reported by the original method.


### Grading
All submissions will be evaluated using:

- **Model:** `Qwen/Qwen3-4B`
- **Evaluation Metric:** Perplexity (PPL)
- **Performance Baseline:** AdaptQuant
- **AdaptQuant PPL:** `13.98`

The score will be based on:

| Item | Weight |
| --- | ---: |
| Reproduction Performance | 15% |
| Report | 5% |
| **Total** | **20%** |

### Reproduction Performance

Regardless of the method you choose, all submissions will be evaluated using the same Qwen3-4B model and PPL evaluation setting.

The corresponding grading scale is:

| PPL | Performance Score |
| ---: | ---: |
| $PPL \leq 15$ | 15 |
| $15 < PPL \leq 16$ | 12 |
| $16 < PPL \leq 17$ | 9 |
| $17 < PPL \leq 18$ | 6 |
| $18 < PPL \leq 19$ | 3 |
| $PPL > 19$ | 0 |

## Report
Your report should include the following:

1. **Method**
   - If you reproduce **AdaptQuant**, you do **not** need to introduce the method again.
   - If you use another method, briefly describe:
     - The method you used
     - Why it satisfies the calibration-free PTQ requirement
     - Any important differences from AdaptQuant

2. **Result**
   - Final PPL

3. **Observation and Discussion**
   - Important observations from your experiment
   - How the quantization method affects model quality
   - Any difficulties encountered during implementation
   - Possible reasons for performance degradation
   - Limitations or possible improvements

The report should focus on your implementation experience and experimental observations rather than simply repeating the content of the paper.

---

## Submission

The `Part3/` folder must contain all files required to reproduce your experiment.

At minimum, submit:

```text
Part3/
├── README.md
├── requirements.txt
├── StudentID_report.pdf
└── ...
```

### `README.md`

The `README.md` is the most important part of the code submission.

It must clearly explain how the TA can:

1. Rebuild the required environment
2. Install all dependencies
3. Run your implementation
4. Evaluate the final PPL
5. Reproduce the result reported in your submission

Please include all required commands and steps.

For example:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
bash run.sh
```

If additional setup is required, it must also be clearly documented in the README.

---

## Overall Submission

Submit all files using the following directory structure:

```text
StudentID.zip/
├── Part1/
│   └── StudentID_Part1.ipynb
│
├── Part2/
│   ├── requirements.txt
│   ├── StudentID_run.sh
│   ├── run_eval.py
│   ├── StudentID_report.pdf
│   └── ...
│
└── Part3/
    ├── README.md
    ├── StudentID_report.pdf
    └── ...
```


## Penalty

Please make sure that your submission follows the required file structure and can be executed correctly.

The following penalties will be applied:

- **Incorrect filename:** -5% for each incorrectly named file.
- **Part 2 cannot be executed:** Part 2 receives **0 points**.
- **Part 3 cannot be executed:** Part 3 receives **0 points**.
- **Late submission:** -20% of the total score for each day late.
- 
:::danger
**Deadline: November 12, 16:30**
:::

