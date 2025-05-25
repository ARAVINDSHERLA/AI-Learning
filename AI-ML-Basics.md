
Here's a structured overview of **basic AI/ML and NLP concepts** that you should understand for interviews, especially if you're targeting **software engineer**, **ML engineer**, or **AI system design** roles:

---

## **1. Basic AI & ML Concepts**

### **AI vs ML vs DL**

* **AI (Artificial Intelligence):** Any system that mimics human intelligence (e.g., rule-based systems, search trees, ML).
* **ML (Machine Learning):** Subset of AI that learns from data.
* **DL (Deep Learning):** Subset of ML using neural networks (e.g., CNNs, RNNs, Transformers).

### **Types of Machine Learning**

* **Supervised Learning:** Labeled data (e.g., regression, classification)
* **Unsupervised Learning:** No labels (e.g., clustering, dimensionality reduction)
* **Semi-supervised / Self-supervised:** Small labeled + large unlabeled data
* **Reinforcement Learning:** Agent learns via trial and error (e.g., games, robotics)

### **Key Algorithms**

* **Linear Regression, Logistic Regression**
* **Decision Trees, Random Forest, XGBoost**
* **KNN, Naive Bayes**
* **SVM (Support Vector Machine)**
* **Clustering:** K-Means, DBSCAN
* **Dimensionality Reduction:** PCA, t-SNE

---

## **2. Deep Learning Basics**

* **Neural Networks (NN):** Layers of neurons with weights and activation functions
* **Activation Functions:** ReLU, Sigmoid, Tanh, Softmax
* **Loss Functions:** MSE, CrossEntropy, Hinge
* **Optimization:** Gradient Descent, Adam, RMSProp
* **Overfitting vs Underfitting:** Bias-variance trade-off
* **Regularization:** Dropout, L1/L2 penalties

---

## **3. Natural Language Processing (NLP) Basics**

### **Core Tasks**

* **Text Classification:** Spam detection, sentiment analysis
* **Named Entity Recognition (NER):** Extract people, places, dates
* **POS Tagging:** Label words with parts of speech
* **Machine Translation:** Translate between languages
* **Question Answering, Summarization, Chatbots**

### **Key Concepts**

* **Tokenization:** Breaking text into words/subwords
* **Stemming vs Lemmatization**
* **Bag of Words, TF-IDF**
* **Word Embeddings:** Word2Vec, GloVe, FastText
* **Transformers & BERT:** Self-attention, context-aware embeddings
* **Sequence Models:** RNNs, LSTMs, GRUs → now mostly replaced by Transformers

---

## **4. Model Evaluation**

* **Classification Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
* **Regression Metrics:** RMSE, MAE, R2-score
* **Confusion Matrix**
* **Cross-validation, Train/Val/Test split**

---

## **5. Production Readiness Concepts**

* **Model Drift / Data Drift**
* **Model Versioning**
* **Model Serving:** FastAPI, Flask, TorchServe, KServe
* **Batch vs Real-Time Inference**
* **Monitoring:** Latency, Throughput, Accuracy in prod
* **CI/CD for ML (MLOps)**

---

## **6. Popular Tools & Libraries**

* **Scikit-learn**: Classical ML
* **Pandas, NumPy**: Data processing
* **TensorFlow, PyTorch**: Deep Learning
* **Transformers (HuggingFace)**: NLP models
* **Keras, Fastai**: High-level DL APIs

---

Would you like a **flashcard-style summary**, **interview Q\&A sheet**, or **hands-on Python notebooks** for these concepts?

**Model as a Service (MaaS)** is a way to **deploy machine learning models** as **independent, scalable services** that can be accessed over a **network via APIs**. It abstracts the model serving infrastructure so consumers can use it like any other web service.

---

### **Key Characteristics of Model as a Service**

| Feature         | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| **Interface**   | Exposes an API (typically REST or gRPC)                             |
| **Decoupling**  | Model is separate from the main application logic                   |
| **Reusability** | Multiple clients or systems can access the same model service       |
| **Scalability** | Service can scale independently based on load (horizontal/vertical) |
| **Versioning**  | Different versions of models can coexist (e.g., `/v1`, `/v2`)       |
| **Monitoring**  | Track latency, throughput, and accuracy drift                       |
| **Security**    | Auth/authz, rate limiting, logging                                  |

---

### **Common Architecture**

```
                +-------------------------+
                |   Client Application    |
                +-------------------------+
                           |
                           |  HTTP/gRPC
                           v
              +--------------------------+
              |     Model API Layer      |   (FastAPI, Flask, gRPC server)
              +--------------------------+
                           |
                           v
         +--------------------------------------+
         |    Model Runtime / Serving Engine    | (TorchServe, TF Serving, etc.)
         +--------------------------------------+
                           |
                           v
                +--------------------+
                |   Trained Model    |
                +--------------------+
```

---

### **Popular Tools for Model as a Service**

| Tool                   | Best For                      | Language    |
| ---------------------- | ----------------------------- | ----------- |
| **FastAPI**            | Lightweight real-time API     | Python      |
| **Flask**              | Simple REST API               | Python      |
| **TorchServe**         | Production PyTorch models     | Python/Java |
| **TensorFlow Serving** | Scalable TensorFlow models    | C++/Python  |
| **KServe (KFServing)** | Kubernetes-native, any model  | Kubernetes  |
| **Seldon Core**        | MLOps and multi-model serving | Kubernetes  |
| **BentoML**            | Simple packaging & deployment | Python      |
| **Amazon SageMaker**   | Fully managed model service   | AWS         |

---

### **Basic Example using FastAPI (PyTorch model)**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextIn):
    inputs = tokenizer(data.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return {"positive": probs[0][1].item(), "negative": probs[0][0].item()}
```

---

### **When to Use Model as a Service**

* You have multiple services consuming predictions
* You need real-time inference
* You want to scale models independently
* You plan to monitor, version, and A/B test models

---

Would you like a **real-world MaaS deployment example (Docker, FastAPI, and TorchServe)** or guidance for **multi-model serving**?

Serving **multiple models** in production—especially with **different types, versions, or use cases**—requires careful planning for **routing, resource management, versioning, and scaling**. Here’s a complete guide:

---

## **1. Multi-Model Serving Approaches**

### **Option A: One Model per Endpoint (FastAPI/Flask)**

* Spin up separate endpoints for each model.
* Good for lightweight models or when deploying via Docker.

**Example:**

```python
@app.post("/predict/model1")
def predict_model1(data: Input):
    ...

@app.post("/predict/model2")
def predict_model2(data: Input):
    ...
```

**Pros:** Simple, fast dev
**Cons:** Memory usage grows, poor resource isolation

---

### **Option B: Dynamic Loading with Model Registry**

* Load models on-demand from disk/S3 using a registry
* Cache active models in memory

```python
MODEL_REGISTRY = {
    "sentiment": "models/sentiment.onnx",
    "toxicity": "models/toxicity.onnx"
}

loaded_models = {}

def get_model(name):
    if name not in loaded_models:
        loaded_models[name] = load_model(MODEL_REGISTRY[name])
    return loaded_models[name]
```

**Pros:** Scalable, memory efficient
**Cons:** Slight latency on first request

---

### **Option C: Use TorchServe / TensorFlow Serving**

* Native support for multi-model via configuration

**TorchServe `config.properties`:**

```ini
model_store=models
load_models=sentiment.mar,emotion.mar
```

**Inference request:**

```bash
curl -X POST http://localhost:8080/predictions/sentiment -T input.json
```

**Pros:** Production-ready, batching, GPU sharing
**Cons:** Needs proper packaging and config

---

### **Option D: KServe (KFServing)**

* Deploy multiple models as **individual Kubernetes services**
* Use **ModelMesh** for dynamic multi-model routing and auto-scaling

**Pros:** Kubernetes-native, GPU efficient, supports any framework
**Cons:** Complex setup, Kubernetes required

---

## **2. Key Best Practices**

### **a. Model Routing**

Use a router layer (e.g., FastAPI, NGINX, or a gateway) to route requests to the correct model:

```http
/predict/sentiment → sentiment model
/predict/toxicity → toxicity model
```

### **b. Model Versioning**

Support versioning like:

```
/v1/sentiment
/v2/sentiment
```

Allow A/B testing and rollback.

### **c. Autoscaling**

* Use Kubernetes HPA or KServe autoscaling
* Monitor GPU/CPU and scale out model services independently

### **d. Caching and Memory Management**

* Limit number of active models
* Unload rarely used ones with LRU eviction
* Quantize models to reduce size

---

## **3. Tooling Recommendations**

| Use Case                      | Tool                    |
| ----------------------------- | ----------------------- |
| Fast prototyping              | FastAPI + dynamic load  |
| Real-time prod                | TorchServe / TF Serving |
| Large-scale, many models      | KServe + ModelMesh      |
| Multi-framework, cloud-native | Seldon Core             |

---

Would you like a **codebase template** for dynamic multi-model loading in FastAPI, or a **deployment guide using TorchServe or KServe**?

Yes, exactly — a **model registry** is a **centralized system** where you can **store, manage, version, and serve machine learning models**. It acts like a **single source of truth** for all models across development, testing, and production environments.

---

## **What a Model Registry Provides**

### **1. Centralized Model Storage**

* All models are stored in one place (e.g., S3, local storage, or database).
* Can store metadata like version, author, metrics, etc.

### **2. Versioning**

* Supports multiple versions of the same model (v1, v2, etc.)
* Rollback, audit trail, and A/B testing become easier.

### **3. Lifecycle Stages**

* Track and move models across:

  * `Staging` → `Production` → `Archived`
  * Approval workflows (e.g., only promote after testing)

### **4. Accessibility**

* Models can be pulled or queried by:

  * Batch pipelines
  * Real-time inference servers (e.g., FastAPI, TorchServe)
  * External systems (via APIs)

---

## **Popular Model Registry Tools**

| Tool                         | Features                           | Storage Backend |
| ---------------------------- | ---------------------------------- | --------------- |
| **MLflow Registry**          | Open-source, lifecycle, versioning | S3, GCS, Azure  |
| **Weights & Biases**         | Hosted, collaborative tracking     | Cloud           |
| **SageMaker Model Registry** | Fully managed on AWS               | S3              |
| **Neptune.ai**               | Model versioning, logging, UI      | Cloud           |
| **DVC**                      | Git-like model tracking            | File system, S3 |
| **KServe + ModelMesh**       | Dynamic multi-model loading        | PVC, MinIO      |

---

## **Example: MLflow Model Registry Flow**

```bash
# Register a model
mlflow models register -m runs:/<run_id>/model -n sentiment-classifier

# Promote to production
mlflow models transition --model-name sentiment-classifier --version 3 --stage Production
```

**From FastAPI or batch job:**

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/sentiment-classifier/Production")
preds = model.predict(input_data)
```

---

## **How to Integrate It**

* Batch pipelines pull latest "Production" model
* Real-time FastAPI server loads models dynamically from the registry
* CI/CD promotes models from staging to prod after validation

---

Would you like a full **MLflow registry setup + FastAPI integration guide** or want to explore **registry design in a custom microservices system**?

