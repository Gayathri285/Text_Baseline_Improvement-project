::: {align="center"}
# Efficient Transformers for Long-Sequence Sentiment Classification

### A Comparative Study of Adaptive CUR-Inspired Landmark Attention and Standard Self-Attention on IMDb

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-IMDb-yellow)](https://huggingface.co/datasets/stanfordnlp/imdb)
[![Task](https://img.shields.io/badge/Task-Sentiment%20Classification-00599C)]()

**Exploring whether efficient attention mechanisms can reduce
computation while preserving sentiment-classification performance.**
:::📌 Project Overview
Standard Transformer self-attention compares every position with every
other position. For a sequence of length (n), this produces an (n
\times {=tex}n) attention-score matrix and quadratic (O(n^2))
scaling.

This project investigates efficient alternatives for long byte-level
IMDb movie reviews. Four Transformer encoders use the same architecture
and differ only in their attention mechanism:
1. Full self-attention --- every valid byte attends to every other
   valid byte.
2. Fixed landmark attention --- each review is represented by up to
   64 segment-mean landmarks.
3. Adaptive landmark attention --- the landmark budget varies with
   the review length, between 16 and 64.
4. CUR attention --- a paper-style CUR approximation reconstructs
   attention using selected rows, columns and an iterative
   pseudo-inverse.
The study is inspired by Drop-in efficient self-attention approximation
method (François et al., 2025). It explores the paper's suggestion of
using a variable number of selected indices per sequence.
🎯 Objectives
- Prepare IMDb reviews as byte-level sequences up to 4,000 bytes.
- Establish a full-attention Transformer baseline.
- Implement fixed and length-adaptive landmark attention.
- Implement paper-style CUR attention with an iterative
  pseudo-inverse.
- Compare classification quality and computational efficiency under a
  shared encoder and training setup.
- Examine whether adaptive landmark selection can retain accuracy with
  fewer representatives.
🗂️ Dataset & Preprocessing
Dataset: IMDb Large Movie Review Dataset (stanfordnlp/imdb on
Hugging Face)

  Item                        Description
  Task                        Binary sentiment classification
Labels                      Negative (0), Positive (1)
Original training reviews   25,000
Training split used         22,500
Validation split            2,500
Test reviews                25,000
Maximum sequence length     4,000 bytes

Preprocessing steps: - Replace HTML line-break tags (<br />) with
spaces. - Convert reviews to lowercase. - Encode text as UTF-8 bytes. -
For reviews longer than 4,000 bytes, retain the first 2,000 and last
2,000 bytes. - Pad shorter reviews; reserve token ID 0 for padding.
🧠 Model Design
All four variants share the same encoder:
- Byte embedding dimension: 64
- Convolution stem: 1-D convolution, kernel size 5, GELU and residual
  connection
- Transformer encoder: 2 layers, 2 attention heads
- Feed-forward dimension: 128
- Dropout: 0.1
- Classification: masked mean and max pooling followed by a two-class
  linear layer
- Trainable parameters: 104,514 per model
Attention variants
  Variant             Attention strategy                                   Landmark budget
  Full                Standard full self-attention                     All valid positions
Fixed landmark      Segment-mean landmarks                                            64
Adaptive landmark   Length-based segment-mean landmarks                           16--64
CUR                 CUR-style reconstruction with pseudo-inverse                      64

For adaptive attention, the number of landmarks is determined from the
valid review length (L):

\(m = `\min`{=tex}(64,`\max`{=tex}(16,L/64))\)

The observed average on the test set was 22.7 landmarks per review.
⚙️ Training Configuration
  Setting                             Value
  Framework                           PyTorch 2.11.0 (CUDA)

  Hardware used                       Tesla T4 GPU (Google Colab)

  Optimizer                           AdamW

  Learning rate                       5e-4

  Weight decay                        0.01

  Batch size                          16

  Maximum epochs                      15

  Learning-rate schedule              6% warmup + cosine decay

  Loss                                Cross-entropy with 0.05 label
smoothing

  Gradient clipping                   1.0

  Early stopping                      4 epochs without validation
improvement

  Random seed                         42
Precision                           Mixed precision (fp16); CUR
pseudo-inverse calculation uses
fp32
📊 Results
Test accuracy
  Attention mechanism     Test accuracy
  Full self-attention        84.01%
Fixed landmark             83.80%
Adaptive landmark          84.06%
CUR attention              84.31%

The largest accuracy difference is only 0.5 percentage points. Each
model was trained once, so these results do not establish a
statistically reliable ranking among the four variants.
Computational efficiency
  Model          Training     Training  Inference /    Inference     Peak GPU
time / epoch     speed-up        batch     speed-up       memory
  Full           190.53 s        1.00×     34.02 ms        1.00×     0.445 GB

  Fixed           42.86 s        4.44×     10.73 ms        3.17×     0.401 GB
landmark                                                       

  Adaptive        42.73 s        4.46×     10.76 ms        3.16×     0.403 GB
landmark                                                       
CUR             82.48 s        2.31×     20.84 ms        1.63×     0.753 GB
Key observations - Fixed and adaptive landmark attention achieved
accuracy close to full attention while training about 4.4× faster and
running inference about 3.2× faster. - Adaptive attention used an
average of 22.7 landmarks, corresponding to about 0.57% of the
full-attention score count in the analytical comparison. - CUR attention
recorded the highest accuracy, but its small advantage is within the
uncertainty of this single-run experiment. - Peak memory was not reduced
in this implementation; the full-attention baseline already uses
PyTorch's fused attention kernel.
🔍 Conclusion
The experiments suggest that representative landmark-based attention can
match full self-attention accuracy on this byte-level IMDb task while
reducing measured training and inference time. Adaptive landmark
attention retained comparable accuracy with fewer landmarks on average.

Within this experiment, simple landmark pooling provided the strongest
balance between accuracy and computational cost. The results should be
interpreted in the context of the single-run setup and this particular
dataset and architecture.
⚠️ Limitations
- Each model was trained only once with seed 42.
- Several preprocessing, architecture and training improvements were
  introduced together, so their individual effects were not isolated.
- Adaptive landmark count is based on sequence length, not learned
  content importance.
- The batched adaptive implementation uses 64 landmark slots with
  masking, so the reduced landmark count did not create additional
  measured wall-clock savings over fixed landmarks.
- Memory reduction was not achieved.
- The study uses a relatively small model and one binary IMDb task;
  other Long Range Arena tasks and longer inputs were not tested.
- Base-paper results use a different experimental setup and are not
  directly comparable.
🚀 Future Work
- Repeat experiments across multiple random seeds.
- Perform landmark-count and component ablation studies.
- Explore content-aware learned landmark selection.
- Bucket reviews by length to better realize adaptive-computation
  savings.
- Test larger encoders and longer sequences.
- Implement fused landmark/CUR attention and evaluate on additional
  sequence tasks.
🧰 Technologies
Python · PyTorch · Hugging Face Datasets · NumPy · IMDb ·
Google Colab · CUDA
▶️ Running the Project
The report identifies Text_Baseline.ipynb as the notebook containing
the end-to-end pipeline.
1. Open Text_Baseline.ipynb in Google Colab or a compatible Jupyter
   environment.
2. Select a CUDA-enabled runtime for the reported GPU training setup.
3. Install or enable the dependencies used by the notebook, including
   PyTorch and Hugging Face Datasets.
4. Run the notebook cells in order to load and preprocess IMDb, train
   the attention variants, and evaluate the models.
The notebook and its exact dependency lockfile should be included in
this repository. This README is based on the project report; adjust
the run instructions if your committed notebook has additional setup
steps.

📚 Reference
François, D., Saillot, M., Klein, J., Bissyandé, T. F., & Skupin, A.
(2025). Drop-in efficient self-attention approximation method. Machine
Learning, 114, 139. https://doi.org/10.1007/s10994-025-06768-3
::: {align="center"}
**Built as an academic project in Transformer Models & Sequence
Learning**

*Learn · Experiment · Compare · Improve*
:::
