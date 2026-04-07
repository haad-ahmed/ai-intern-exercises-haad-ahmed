# transformer-day1-exercise

## 1️⃣ What is Generative AI?

Generative AI refers to a class of machine learning models that can produce new content — text, images, audio, code, or video — by learning the underlying patterns and distributions of training data. Rather than simply classifying or predicting a label, generative models learn *how data is structured* and use that understanding to generate novel, coherent outputs.

**How it differs from traditional machine learning:**

Traditional ML models are discriminative — they map inputs to outputs (e.g., "is this email spam or not?"). They do not create anything new; they draw boundaries. Generative AI, by contrast, models the data distribution itself. It can produce outputs that never existed in the training set, making it fundamentally creative rather than merely classificatory.

**3 Real-World Applications:**

1. **Code generation** — Tools like GitHub Copilot use LLMs to suggest and complete code in real time, dramatically accelerating software development.
2. **Medical imaging** — Generative models synthesize realistic medical scans to augment training datasets, helping build better diagnostic tools where real data is scarce.
3. **Conversational assistants** — LLM-powered chatbots handle customer support, tutoring, and general Q&A at scale with human-like language understanding.

---

## 2️⃣ Self-Attention Explained

**Sentence:** `"The cat sat on the mat"`

### Query (Q), Key (K), and Value (V)

For each token in the sentence, three vectors are derived via learned linear projections:

- **Query (Q):** Represents what the current token is *looking for* — its information need.
- **Key (K):** Represents what each token *offers* — a description of its content.
- **Value (V):** The actual information a token *contributes* once it is deemed relevant.

The attention score between two tokens is computed as the dot product of one token's Query with another token's Key. For the word "sat", its Query would score highly against Keys of "cat" (the subject) and "mat" (the location), allowing the model to pull in both contexts via their Values.

### Why scale by √d_k?

The dot product Q·Kᵀ grows in magnitude as the dimension `d_k` increases, pushing values into regions where the softmax function produces near-zero gradients (the saturation zone). Dividing by √d_k keeps the scores in a moderate range, preserving healthy gradient flow during training.

### Why apply Softmax?

Softmax converts the raw attention scores into a probability distribution that sums to 1. This means each token attends to all others with non-negative weights that represent *relative importance*. It makes the attention mechanism differentiable and interpretable as a weighted combination.

### What problem does attention solve that RNNs struggled with?

RNNs process sequences step by step, so information from early tokens must travel through many hidden states to influence later ones — leading to the **vanishing gradient problem** and poor long-range dependency modeling. Self-attention connects every token to every other token in a single operation regardless of distance, solving long-range dependency directly and enabling full parallelization.

---

## 3️⃣ Encoder vs Decoder Comparison

| Component | Encoder | Decoder |
|---|---|---|
| **Primary role** | Understands and encodes input | Generates output token by token |
| **Self-attention type** | Bidirectional (attends to all tokens) | Masked (attends only to past tokens) |
| **Cross-attention** | Not present | Attends to encoder output |
| **Masked attention** | Not used | Used to prevent seeing future tokens during training |
| **Typical use cases** | Classification, NER, embeddings (e.g., BERT) | Text generation, translation (e.g., GPT, T5 decoder) |

**Masked attention** ensures that during training, the decoder cannot "cheat" by looking at future tokens — each position can only attend to positions before it.

**Cross-attention** allows the decoder to query the encoder's output, aligning generated tokens with relevant parts of the input (essential in translation, summarization).

---

## 4️⃣ Vision Transformers (ViT) — High-Level Explanation

A Vision Transformer applies the transformer architecture — originally designed for text — directly to images by treating image regions as tokens.

**Image patches:** An input image (e.g., 224×224 pixels) is divided into a grid of fixed-size patches (e.g., 16×16 pixels each), yielding 196 patches for that configuration.

**Patches to tokens:** Each patch is flattened into a 1D vector and passed through a linear projection layer, producing a patch embedding of fixed dimension `d`. A learnable `[CLS]` token is prepended to the sequence, and its final hidden state is used for classification.

**Why positional embeddings are necessary:** Unlike CNNs, transformers have no built-in notion of spatial order. Positional embeddings (either learned or fixed sinusoidal vectors) are added to each patch embedding so the model knows where in the image each patch came from. Without them, the model would treat patches as an unordered set.

**How ViT differs from CNNs conceptually:** CNNs use local convolution filters that slide across the image, building up spatial hierarchies from edges to textures to objects — inductive biases baked into the architecture. ViT makes no such assumption. It treats every patch as equally distant from every other patch at the start, and learns global relationships through self-attention from the very first layer. This makes ViT more flexible but requires significantly more data to learn spatial structure that CNNs get for free.
