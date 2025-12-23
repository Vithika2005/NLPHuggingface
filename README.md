# NLP & Huggingface
What are we gonna build? BERT-Sentiment Analysis, T5-News summarizer

##  What Are We Gonna Build?

### Project A: BERT – Sentiment Analysis  
- Task: Binary sentiment classification  
- Model: **BERT (Encoder-only Transformer)**  
- Dataset: IMDb  
- Output: Positive / Negative sentiment  

### Project B: T5 – News Summarization  
- Task: Abstractive text summarization  
- Model: **T5 (Encoder-Decoder Transformer)**  
- Dataset: CNN/DailyMail  
- Output: Concise summary of news articles  

---

# Standard NLP Pipeline
Text->Tokens->IDS->Vectors->Model->Prediction
- **Tokenization**: Convert text → tokens  
- **Embeddings**: Convert token IDs → dense vectors

# Tokenization
### 1️⃣ WordPiece (Used By BERT)
- Breaks unknown words into known sub-tokens  
- Minimizes `[UNK]` tokens (Unknown words to known chunks).
- Designed for **Masked Language Models (MLM)**  
**Example:**  
`unhappiness → un + ##happy + ##ness`

### 2️⃣ Byte Pair Encoding (BPE – GPT style)
- Starts from characters  
- Merges most frequent character pairs  
- Frequency-based compression approach
- Based on frequency for statistical compression 

### 3️⃣ SentencePiece (T5)
- Language-agnostic  
- No dependency on whitespace  
- Robust for multilingual data  
---

# Embeddings
1. Manual Embedding: 1 word to 1 fixed vector.
- Vectors feed into RNN, CNN, LSTM, all classical ML model. They are explicit in nature. Even used in Non neural models. 
Tokenization->Word2Vec->LSTM/GRU/CNN->Classifer
- FastText: Fed into linear classification.
- TF-IDF: Weighted embedding
WordVectors->Pooling->FeatureVector->ML model

| Method | Description |
|------|------------|
| Word2Vec | One word → one fixed vector |
| GloVe | Global word co-occurrence |
| FastText | Subword embeddings (handles misspellings) |
| TF-IDF | Weighted sparse vectors |


2. Transform NLP (Automatic Embeddings):
- Used with BERT/T5/GPT/Llama
- Models will have: Token Embeddings, Position Embeddings, Segment Embeddings. (For BERT)
- Tokenization->InputID->InternalEmbeddingLayer->TransformLayer

STATIC EMBEDDINGS:
1. 1 Word-> 1 Vector
2. No position
3. External
4. Shallow Semantic

TRANSFORM EMBEDDINGS:
1. Contextual Vector
2. Position aware
3. Internal
4. Deep Semantic 

# Transformer Architecture
1. BERT: Encoder Only. Used for Sentiment analysis, classifiaction. Pretraining: Masked Language Modelling (MLM), Next Sentence Prediction (NSP).
2. T5: Encoder-Decoder. Input is text and output is text. Used for summarization. Pretraining: Seq2Seq learning, Span Corruption.

# Hugging Face
1. Loading pretrained model for Tokenizer, model weights trained on massive corpus.
2. Fine Tuning: Tuning weights. BERT for Sentiment Analysis, T5 Summarization.
3. LoRA (Learns the fine tuning to change weights):
LoRA:Freeze base model, train small low rank matrices. It believes most task specific changes live in small subspace. Therefore we don't need to change everything, just change in small directions.
PEFT (Parameter Efficient Fine Tuning) includes LoRA (most popular) (Low Rank Adaptor). Its prefix tuning prompt tuning adapter. 
