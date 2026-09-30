# AI Engineering: Hands-on
![Stars](https://img.shields.io/github/stars/Ramakm/ai-hands-on?style=flat-square)
![Forks](https://img.shields.io/github/forks/Ramakm/ai-hands-on?style=flat-square)
![PRs](https://img.shields.io/github/issues-pr/Ramakm/ai-hands-on?style=flat-square)
![Issues](https://img.shields.io/github/issues/Ramakm/ai-hands-on?style=flat-square)
![Contributors](https://img.shields.io/github/contributors/Ramakm/ai-hands-on?style=flat-square)
![License](https://img.shields.io/github/license/Ramakm/ai-hands-on?style=flat-square)
<img width="2000" height="600" alt="image" src="https://github.com/user-attachments/assets/0c161d9a-3227-4dd7-a416-ad97bc9742a9" />

A complete, hands-on guide to becoming an AI Engineer.

This repository is designed to help you learn AI from first principles, build real neural networks, and understand modern LLM systems end-to-end.
You'll progress through math, PyTorch, deep learning, transformers, RAG, and OCR — with clean, intuitive Jupyter notebooks guiding you at every step.

Whether you're a beginner or an engineer levelling up, this repo gives you the clarity, structure, and intuition needed to build real AI systems.

#### ⭐ Star This Repo

If you learn something useful, a star is appreciated.

## Repository Structure

### 1. Math Fundamentals
- Math functions, derivatives, vectors, and gradients
- Matrix operations and linear algebra
- Probability and statistics

### 2. PyTorch Basics
- Creating and manipulating tensors
- Matrix multiplication, transposing, and reshaping
- Indexing, slicing, and concatenating tensors
- Special tensor creation functions

### 3. Neural-Network(NN)
- Building neurons, layers, and networks from scratch
- Normalization techniques (RMSNorm)
- Activation functions
- Optimizers (Adam, Muon) and learning rate decay

### 4. Transformers
- Attention and self-attention mechanisms
- Multi-head attention
- Decoder-only transformer architecture

### 5. Retrieval-Augmented Generation (RAG)
- Building RAG pipelines end to end
- Indexing, retrieval, chunking strategies
- Integrations with embedding models and vector stores
- Cloud LLM support: [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=ai-hands-on) (`deepseek-ai/DeepSeek-V3-0324` by default), [MiniMax](https://www.minimax.io/) (M3), OpenAI, or any OpenAI-compatible API for richer answer generation

### 6. Optical Character Recognition (OCR)
- OCR pipeline and utilities
- Preprocessing images and extracting text

### 7. Additional Modules
Beyond the numbered learning track, the repository also contains the following folders:

- **LLM**: LLM fundamentals, tokens, and token embeddings
- **Fine-Tuning**: LoRA, QLoRA, and Unsloth fine-tuning notebooks
- **Basic ML Model Implementation**: Classical supervised and unsupervised ML notebooks
- **DL**: Deep learning projects with production-style serving (FastAPI / Streamlit)
- **Brazil-Ecommerce-Data-Analysis**: Exploratory data analysis on the Olist dataset
- **YoutubeMCP**: A minimal MCP server for fetching YouTube transcripts
- **Interview Prep, Books, Data, Start_here**: Supporting resources and sample data

## Project Structure

A detailed map of the repository:

```text
ai-hands-on/
├── README.md                         # This file
├── LICENSE                           # MIT License
├── requirements.txt                  # Root Python dependencies
├── ANN_&_CNN.ipynb                   # ANN vs CNN on Fashion-MNIST (root copy)
├── ATLAS_CLOUD_PROVIDER_REVIEW.md    # Notes on the Atlas Cloud provider integration in 5.RAG
├── test.txt                          # Sample network-intrusion CSV data
│
├── .github/
│   └── FUNDING.yml                   # Sponsorship config
│
├── Start_here/
│   └── learning_path.md              # Recommended step-by-step learning path
│
├── 0.1-Do-you-know/                  # Short "did you know" explainers
│   ├── ANN(Artificial_Neural_Network_).ipynb
│   └── Efficient_Iteration_In_Pandas.ipynb
│
├── 1.Math/                           # Math fundamentals
│   ├── 1. Math Functions.ipynb
│   ├── 2. Partial Derivatives.ipynb
│   ├── 3. Vectors.ipynb
│   ├── 4. Gradient.ipynb
│   ├── 5. Metrics.ipynb
│   └── 6. Probability.ipynb
│
├── 2.Pytorch/                        # PyTorch tensor basics
│   ├── 1. Tensor.ipynb
│   ├── 2. Matrix Multiplication.ipynb
│   ├── 3. Tensor Transposing.ipynb
│   ├── 4. Reshaping tensor.ipynb
│   ├── 5. Indexing & Slicing.ipynb
│   ├── 6. Tensor Concatenating.ipynb
│   └── 7. Linspace and Arrange Tensor.ipynb
│
├── 3.Neural-Network(NN)/             # Neural networks from scratch
│   ├── 1.Single Neuron.ipynb
│   ├── 2. Layer building.ipynb
│   └── 3. BackPropagation.md
│
├── 4.Transformer/                    # Transformer building blocks
│   ├── 1. Attention Mechanism.ipynb
│   ├── 2. Self Attention.ipynb
│   ├── 3. Multi-head Attention.ipynb
│   └── 4. Decoder.ipynb
│
├── 5.RAG/                            # Retrieval-Augmented Generation (cybersecurity use case)
│   ├── README.md
│   ├── .env.example                  # API key / provider configuration template
│   ├── requirements.txt
│   ├── data/processed_texts/         # Source documents (firewall, incident response, vulnerability scan)
│   ├── embeddings/faiss_index/       # Prebuilt FAISS index + metadata
│   ├── src/
│   │   ├── extract_text.py           # Text extraction
│   │   ├── create_embeddings.py      # Build embeddings and FAISS index
│   │   ├── retrieve_context.py       # Semantic retrieval
│   │   ├── generate_answer.py        # Answer generation
│   │   ├── llm_provider.py           # Provider abstraction (Atlas Cloud, MiniMax, OpenAI, compatible APIs)
│   │   └── app.py                    # Application entry point
│   └── tests/                        # test_integration.py, test_llm_provider.py
│
├── 6.OCR/                            # Optical Character Recognition
│   ├── README.md
│   ├── main.py                       # API / app entry point
│   ├── utils.py                      # Image preprocessing and OCR helpers
│   ├── pytess_gemini.ipynb           # Pytesseract + Gemini notebook
│   ├── render.yaml                   # Render deployment config
│   ├── requirements.txt
│   └── sample_data/                  # Sample images (incl. a rotated one)
│
├── LLM/                              # LLM fundamentals
│   ├── 1_LLM_Fundamentals_.ipynb
│   └── 2_Tokens_and_Token_Embeddings.ipynb
│
├── Fine-Tuning/                      # Parameter-efficient fine-tuning
│   ├── Fine_Tuning_a_JSON_File_Using_Unsloth_&_LoRA.ipynb
│   ├── Lora-Implementation.ipynb
│   └── Qlora-Implementation.ipynb
│
├── Basic ML Model Implementation/    # Classical ML notebooks
│   ├── 1_Base_ML_Model_Linear_&_Ridge_Regression.ipynb
│   ├── LM_Implementation.ipynb
│   ├── Logistic_Regression_Model_Implementation.ipynb
│   ├── Decision_Tree_Model_Implementation.ipynb
│   ├── Naïve_Bayes_Classification_Implementation_.ipynb
│   ├── Support_Vector_Machine_[SVM]_Classification_Implementation_.ipynb
│   ├── Classification_Models_Implementation[Supervised].ipynb
│   └── Unsupervised_Learning_Models_Implementation.ipynb
│
├── DL/                               # Deep learning projects (notebook -> production)
│   ├── ANN-CNN Prod/                 # Fashion-MNIST ANN vs CNN served with FastAPI
│   │   ├── train.py, config.py
│   │   ├── app/                      # FastAPI app, preprocessing, static upload page
│   │   └── models/                   # Trained .keras models + metrics.json
│   ├── RNN-LSTM Prod/                # NumPy RNN vs LSTM from scratch + FastAPI + web UI
│   │   ├── seqnet/                   # Models, optimizers, training, persistence, API
│   │   ├── web/                      # Frontend (HTML/CSS/JS)
│   │   ├── artifacts/                # Trained weights
│   │   ├── tests/                    # Gradient checks, API, notebook sync tests
│   │   └── Dockerfile, Makefile, start.sh
│   └── RNN-LSTM-GRU/                 # HAR sensor data: RNN vs LSTM vs GRU
│       ├── backend/                  # Config, data, model, training, FastAPI
│       ├── frontend/app.py           # Streamlit UI
│       └── artifacts/                # Trained models, scaler, encoder, results
│
├── Brazil-Ecommerce-Data-Analysis/   # Olist e-commerce EDA
│   ├── README.md
│   └── customer / geolocation / order_items / order_payments /
│       order_reviews / orders / products / sellers _analysis.ipynb
│
├── YoutubeMCP/                       # MCP server for YouTube transcripts
│   ├── README.md, Project.md
│   ├── main.py, server.py, test.py
│   ├── pyproject.toml
│   └── src/                          # service.py, utils.py
│
├── Data/                             # Sample datasets
│   ├── json_extraction_dataset_500.json   # Used for JSON-extraction fine-tuning
│   └── test.txt
│
├── Interview Prep/
│   └── AI Engineering Topics To Prepare Must.html
│
└── Books/                            # Reference PDFs (SQL, AI Engineering, Deep Learning, ML)
```

### Folder Guide

| Folder | Type | What you will find |
| ------ | ---- | ------------------ |
| `Start_here/` | Guide | Learning path for going through the repo in order |
| `0.1-Do-you-know/` | Notebooks | Quick explainers on ANNs and efficient Pandas iteration |
| `1.Math/` | Notebooks | Functions, partial derivatives, vectors, gradients, metrics, probability |
| `2.Pytorch/` | Notebooks | Tensor creation, matmul, transpose, reshape, indexing, concatenation, linspace/arange |
| `3.Neural-Network(NN)/` | Notebooks + Notes | Single neuron, layer building, backpropagation notes |
| `4.Transformer/` | Notebooks | Attention, self-attention, multi-head attention, decoder |
| `5.RAG/` | Project | FAISS-based RAG pipeline with pluggable LLM providers and tests |
| `6.OCR/` | Project | OCR service with preprocessing utilities and Render deployment |
| `LLM/` | Notebooks | LLM fundamentals, tokens, and embeddings |
| `Fine-Tuning/` | Notebooks | LoRA, QLoRA, and Unsloth fine-tuning |
| `Basic ML Model Implementation/` | Notebooks | Regression, logistic regression, trees, Naive Bayes, SVM, unsupervised learning |
| `DL/` | Projects | ANN/CNN, RNN/LSTM, and RNN/LSTM/GRU with APIs and UIs |
| `Brazil-Ecommerce-Data-Analysis/` | Notebooks | EDA on customers, geolocation, orders, payments, reviews, products, sellers |
| `YoutubeMCP/` | Project | MCP server that fetches YouTube transcripts |
| `Data/` | Data | Sample datasets |
| `Interview Prep/` | Resource | AI engineering interview topics checklist |
| `Books/` | Resource | Reference PDFs |

## Books

Recommended reading to deepen your understanding (not included):

- `AI Engineering` by Chip Huyen
- `Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow` by Aurélien Géron
- `Deep Learning` by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- `The Elements of Statistical Learning` by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
- `Neural Networks and Deep Learning` by Michael Nielsen
- `SQL Cookbook` by Anthony Molinaro

For more books in AI/ML, I have created another repo for this [Check Here](https://github.com/Ramakm/AI-ML-Book-References.git). I will be adding lot more in coming days/months. If you are interested to read book, go check this repo out.

## Learning Path

For a recommended step-by-step progression through the materials, see the Learning Path:

- `Start_here/learning_path.md`

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Some subfolders (for example `5.RAG/` and `6.OCR/`) include their own `requirements.txt` with additional dependencies.

## Usage

Recommended workflow:

1. Open Jupyter in the project root:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Work through notebooks in order:
   - `1.Math/`
   - `2.PyTorch/`
   - `3.Neural-Network(NN)/`
   - `4.Transformer/`

3. Folder to run separately:
   - `5.RAG/`
   - `6.OCR/`
   - `DL/ANN-CNN Prod/`, `DL/RNN-LSTM Prod/`, `DL/RNN-LSTM-GRU/` (each has its own README and requirements)
   - `YoutubeMCP/` (requires Python 3.12+ and `uv`)
  
4. Resources
5. Basic ML Model Implementation (Supervised + Un-supervised + RL)
   - `1.Linear Regression`
   - `2.Logistic Regression`
   - `3.Decision Tree Model`
   - `4.Naive Bayes Classification`
  
## Machine Learning Frameworks

| Tool         | Category          | Link                                                                                     |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------- |
| Scikit-learn | Traditional ML    | [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)                     |
| XGBoost      | Gradient Boosting | [https://xgboost.ai/](https://xgboost.ai/)                                               |
| LightGBM     | Gradient Boosting | [https://lightgbm.readthedocs.io/en/stable/](https://lightgbm.readthedocs.io/en/stable/) |
| CatBoost     | Gradient Boosting | [https://catboost.ai/](https://catboost.ai/)                                             |


## GEN AI References

| Resource                              | Focus Area             | Link                                                                                                                                                                   |
| ------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Microsoft Generative AI for Beginners | Intro to GenAI         | [https://github.com/microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)                                                   |
| Generative AI for Everyone            | Non-technical overview | [https://www.coursera.org/learn/generative-ai-for-everyone](https://www.coursera.org/learn/generative-ai-for-everyone)                                                 |
| Building Blocks of Generative AI      | Conceptual foundations | [https://shriftman.substack.com/p/the-building-blocks-of-generative](https://shriftman.substack.com/p/the-building-blocks-of-generative)                               |
| The Illustrated Transformer           | Transformers           | [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)                                                             |
| LLMs Explained Briefly                | LLM basics video       | [https://www.youtube.com/watch?v=LPZh9BOjkQs](https://www.youtube.com/watch?v=LPZh9BOjkQs)                                                                             |
| Intro to LLMs                         | LLM overview video     | [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g)                                                                             |
| Understanding LLMs                    | Deep dive              | [https://magazine.sebastianraschka.com/p/understanding-large-language-models](https://magazine.sebastianraschka.com/p/understanding-large-language-models)             |
| Visual Guide to Reasoning LLMs        | Reasoning models       | [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)         |
| Understanding Reasoning LLMs          | Reasoning theory       | [https://magazine.sebastianraschka.com/p/understanding-reasoning-llms](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)                           |
| Understanding Multimodal LLMs         | Vision + text models   | [https://magazine.sebastianraschka.com/p/understanding-multimodal-llms](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)                         |
| Visual Guide to MoE                   | Mixture of Experts     | [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) |
| Finetuning LLMs                       | Model training         | [https://magazine.sebastianraschka.com/p/finetuning-large-language-models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)                   |
| How Transformer LLMs Work             | Architecture           | [https://www.deeplearning.ai/short-courses/how-transformer-llms-work/](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)                           |
| Build GPT from Scratch                | Hands-on               | [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)                                                                             |
| LLM Course (GitHub)                   | Structured learning    | [https://github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course)                                                                                       |
| LLM Course (Hugging Face)             | Practical LLMs         | [https://huggingface.co/learn/llm-course/chapter1/1](https://huggingface.co/learn/llm-course/chapter1/1)                                                               |
| Awesome LLM Apps                      | Project ideas          | [https://github.com/Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)                                                                   |
| How RAG Enhances LLMs                 | RAG                    | [https://awesomeneuron.substack.com/p/how-rag-enhances-llms-a-step-by-step](https://awesomeneuron.substack.com/p/how-rag-enhances-llms-a-step-by-step)                                                       |
| Visual Guide to AI Agents             | AI Agents              | [https://awesomeneuron.substack.com/p/a-visual-guide-to-ai-agents](https://awesomeneuron.substack.com/p/a-visual-guide-to-ai-agents)                                        |

## Contributing

Contributions are welcome!

Please ensure:

- Notebooks are clean (Restart & Run All before committing)
- Existing structure & naming conventions are followed
- PRs are focused, readable, and documented
- In folders like RAG and OCR, please maintain the cleaned structure part
- If you want to add something new folders, make it proper structure way.

## License

- This project is licensed under the MIT License. See `LICENSE` for details.

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)
