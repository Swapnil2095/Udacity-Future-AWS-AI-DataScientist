#!/usr/bin/env python
# coding: utf-8

# # SentimentScope: Sentiment Analysis using Transformers!
# ## Introduction <a name = "introduction"></a>
# 
# In this notebook, you will train a transformer model from scratch to perform sentiment analysis on the IMDB dataset. You are a Machine Learning Engineer at Cinescope, a growing entertainment company working to enhance its recommendation system. Your task is to fine-tune a transformer-based model for sentiment analysis using the IMDB dataset. By classifying reviews as positive or negative, you will help the company better understand user sentiment and deliver more personalized experiences.
# 
# By completing this project, you will demonstrate your competency in the following learning objectives:
# 
# - Load, explore, and prepare a text dataset for training a transformer model using PyTorch.
# - Customize the architecture of the transformer model for a classification task.
# - Train and test a transformer model on the IMDB dataset.
# 
# Now that you have an overview of what you will achieve in this project, let’s move on to the project outline.
# 
# ---
# 
# ### Project Outline
# 
# This notebook is organized into the following sections:
# 
# 
# 1. [Introduction](#introduction): Overview of the project, learning objectives, and understanding sentiment analysis.
# 2. [Load, Explore, and Prepare the Dataset](#load-explore-and-prepare-the-dataset): Load the IMDB dataset, explore it with visualizations, and split it into training and validation sets.
# 3. [Implement a DataLoader in PyTorch](#implement-a-dataloader-in-pytorch): Create the `IMDBDataset` class and use it with the PyTorch `DataLoader`, including tokenization.
# 4. [Customize the Transformer Architecture](#customize-the-transformer-architecture): Modify the transformer model for binary classification.
# 5. [Implement Accuracy Calculation Method](#implement-accuracy-calculation-method): Create a function to compute accuracy for monitoring performance.
# 6. [Train the Model](#train-the-model): Complete and execute the training loop for binary classification.
# 7. [Test the Model](#test-the-model): Evaluate the model on the test dataset and ensure it achieves over 75% accuracy.
# 8. [Conclusion](#conclusion): Summarize the project results and key takeaways.
# 
# Click on the section titles above to navigate directly to the corresponding part of the notebook!
# 
# ---
# 
# Now that we've outlined the structure and objectives of this project, let's delve into the core concept: sentiment analysis.
# 
# ### Understanding Sentiment Analysis
# 
# Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a piece of text. This can range from identifying the polarity (positive, negative, or neutral) of a review to analyzing emotions and opinions.
# 
# In this project, sentiment analysis is explicitly framed as a **binary classification task**, where the goal is to determine whether a given movie review is *positive* or *negative*. This task is central to many real-world applications, including customer feedback analysis, social media monitoring, and recommendation systems. By developing a transformer-based model, you will classify IMDB movie reviews as positive or negative to tackle the challenge faced by your entertainment company CineScope by enhancing its recommendation system, enabling more accurate and personalized suggestions. 
# 
# Reviews labeled as positive will be marked as 1 in the dataset, while negative reviews will be labeled as 0.
# 
# For example, consider the following movie review:
# 
# > "The movie was a rollercoaster of emotions, and I loved every moment of it!"
# 
# This review is clearly positive as it expresses enjoyment and satisfaction with the movie, hence it will be labelled as *positive* or 1 in the dataset. In contrast:
# 
# > "The plot was predictable, and the acting was subpar. A waste of time."
# 
# This review conveys a negative sentiment, criticizing both the plot and acting, hence it will be labelled as *negative* or 0 in the dataset.
# 
# While transformers are often used for generation tasks, they can also be adapted for classification tasks with some modifications to their architecture. You might already be familiar with the tweaks that we will implement in this project.
# 
# 
# ---
# 
# ### Data Description
# 
# The dataset used in this project is the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/), provided in the `aclIMDB_v1.tar.gz` file. Upon extracting the file, you will find the following folder structure:
# 
# ```
# aclIMDB/
# ├── train/
# │   ├── pos/    # Positive reviews for training
# │   ├── neg/    # Negative reviews for training
# │   ├── unsup/  # Unsupervised data (not used in this project)
# ├── test/
# │   ├── pos/    # Positive reviews for testing
# │   ├── neg/    # Negative reviews for testing
# ```
# 
# - **train/**: Contains labeled data for training the model. Reviews in the `pos/` folder should be labeled as positive (1), while reviews in the `neg/` folder should be labeled as negative (0).
# - **test/**: Contains labeled data for evaluating the model. Similar to the training data, `pos/` and `neg/` contain positive and negative reviews, respectively.
# - **unsup/**: Contains unlabeled reviews that are not used in this project.
# 
# Understanding the folder structure is crucial as it guides how we load and preprocess the data for the sentiment classification task.
# 
# ---
# 

# ## <a name="load-explore-and-prepare-the-dataset"></a>Load, Explore, and Prepare the Dataset
# 
# ### 1. Load the Dataset
# The dataset is already available in the environment as `aclIMDB_v1.tar.gz`. We will load it into Pandas DataFrames for easy exploration and preparation.
# 

# In[119]:


import os
import pandas as pd


# In[120]:


# Unpack the dataset - uncomment the line below to run
get_ipython().system('tar -xzf aclImdb_v1.tar.gz')


# In[121]:


pwd


# In[122]:


get_ipython().system('ls')


# In[123]:


print(os.listdir())            # should show 'aclImdb'
print(os.listdir('aclImdb'))


# You have successfully extracted the folder. Go back to your workspace and explore the folder structure and find the relative paths for each of the following:
# - Training positive reviews
# - Training negative reviews
# - Testing positive reviews
# - Testing negative reviews
# 
# Assign the paths of these folders relative to the starter file in the variables below.
# 

# In[124]:


# Define paths to dataset
train_pos_path = 'aclImdb/train/pos' # Path to the directory containing positive reviews from the training set
train_neg_path = 'aclImdb/train/neg' # Path to the directory containing negative reviews from the training set
test_pos_path = 'aclImdb/test/pos' # Path to the directory containing positive reviews from the test set
test_neg_path = 'aclImdb/test/neg' # Path to the directory containing negative reviews from the test set


# Now, you will implement the `load_dataset()` function, which reads all text files in a specified folder and returns their content as a list of strings. This function is essential for loading and preprocessing the dataset in subsequent steps.
# 
# To implement this function:
# 
# 1. **Use the `os` module**: Leverage Python's `os` module to list all files in the folder.
# 2. **Handle file paths**: Use `os.path.join()` to construct full paths for files, ensuring compatibility across operating systems.
# 3. **Read file content**: Open each file in read mode (`'r'`) using UTF-8 encoding to handle text properly.
# 4. **Aggregate results**: Append the content of each file to a list and return it.
# 
# ### Key Points to Consider:
# - Ensure that the function only processes text files (you may use file extensions for filtering if needed).
# - Refer to the [os.listdir documentation](https://docs.python.org/3/library/os.html#os.listdir) for listing files in a directory.
# 

# In[125]:


def load_dataset(folder):
    """
    Reads all text files in the specified folder and returns their content as a list.

    Args:
        folder (str): Path to the folder containing text files.

    Returns:
        list: A list of strings, where each string is the content of a text file.
    """
    #TODO: Implement the function here
    reviews = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                reviews.append(f.read())
    return reviews


# Use the function now to load the training and testing data:

# In[126]:


# Load training and testing data
train_pos = load_dataset(train_pos_path)
train_neg = load_dataset(train_neg_path)
test_pos = load_dataset(test_pos_path)
test_neg = load_dataset(test_neg_path)


# We can convert the data into pandas dataframes to make handling the datasets easier.

# In[127]:


# Create DataFrames
train_df = pd.DataFrame({
    'review': train_pos + train_neg,
    'label': [1] * len(train_pos) + [0] * len(train_neg)
})

test_df = pd.DataFrame({
    'review': test_pos + test_neg,
    'label': [1] * len(test_pos) + [0] * len(test_neg)
})

print(train_df.head())


# You can ensure that your datasets have loaded correctly by running the following code cell. No output means success!

# In[128]:


# Assert that both datasets have the expected number of rows
assert train_df.shape[0] == 25000, "Training dataset does not have 25000 rows."
assert test_df.shape[0] == 25000, "Testing dataset does not have 25000 rows."

# Assert that both datasets have exactly two columns
assert train_df.shape[1] == 2, "Training dataset does not have exactly 2 columns."
assert test_df.shape[1] == 2, "Testing dataset does not have exactly 2 columns."


# ### 2. Explore the Dataset
# Exploration helps us understand the dataset's structure and distribution.
# 
# Here are some suggestions for exploration and visualizations:
# - **Dataset Overview**: Use `DataFrame.info()` and `DataFrame.describe()` to understand the dataset structure and basic statistics.
# - **Label Distribution**: Create bar charts to visualize the number of positive and negative reviews.
# - **Review Length Analysis**: Compute and plot the distribution of review lengths (e.g., number of characters or words).
# - **Sample Reviews**: Print a few positive and negative reviews to understand the text content.
# 
# Write code to explore the dataset in this section.

# In[129]:


# --- Dataset Overview ---
print("Training Data Overview:")
print(train_df.info())
print(train_df.describe())

print("\nTesting Data Overview:")
print(test_df.info())
print(test_df.describe())


# In[130]:


# --- Label Distribution ---
import matplotlib.pyplot as plt

train_label_counts = train_df['label'].value_counts()
test_label_counts = test_df['label'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(train_label_counts.index, train_label_counts.values)
axes[0].set_title("Training Label Distribution")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Negative", "Positive"])
axes[0].set_ylabel("Count")

axes[1].bar(test_label_counts.index, test_label_counts.values)
axes[1].set_title("Testing Label Distribution")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Negative", "Positive"])
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


# In[131]:


# --- Review Length Analysis ---
train_df['review_length'] = train_df['review'].apply(lambda x: len(x.split()))
test_df['review_length'] = test_df['review'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
plt.hist(train_df['review_length'], bins=50, alpha=0.6, label='Train')
plt.hist(test_df['review_length'], bins=50, alpha=0.6, label='Test')
plt.title("Distribution of Review Lengths (in words)")
plt.xlabel("Review length (words)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[132]:


# --- Sample Reviews ---
print("\nSample Positive Review:\n")
print(train_df[train_df['label'] == 1]['review'].iloc[0][:500])

print("\nSample Negative Review:\n")
print(train_df[train_df['label'] == 0]['review'].iloc[0][:500])


# ### 3. Prepare the Dataset
# We will split the training data further into training and validation subsets. The way we constructed the dataset, reviews with positive and negative labels are segregated. To ensure that the validation dataset works well, we first need to shuffle the dataset.
# 

# In[133]:


# Split train data into training and validation sets manually
train_size = int(0.9 * len(train_df))
# Shuffle the dataset
shuffled_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = shuffled_df.iloc[:train_size]
val_data = shuffled_df.iloc[train_size:]


# ### 4. Testing the Tokenizer
# 
# #### Subword Tokenization
# In earlier tasks, you might have encountered character-level tokenization, where each character in the text is treated as a token. While this is straightforward, it is less efficient and may result in larger input sizes, impacting the performance of transformer models.
# 
# To address this, we will use Hugging Face's `AutoTokenizer`, a robust and efficient class designed for tokenizing text based on pretrained models. Specifically, we will utilize the `bert-base-uncased` tokenizer, which applies **subword tokenization**. Subword tokenization involves two steps:
# 
# 1. **Subword Splitting**: Words are split into smaller components (subwords) based on a predefined vocabulary. For example:
#    - Input: "unhappiness"
#    - Subword Splits: `['un', 'happiness']`
# 
# 2. **Token Conversion**: Each subword is then converted into a numerical token ID. For example:
#    - Subword Splits: `['un', 'happiness']`
#    - Token IDs: `[1011, 24123]` (values are illustrative and depend on the tokenizer vocabulary).
# 
# #### About the `bert-base-uncased` Tokenizer
# The `bert-base-uncased` tokenizer, developed by Google researchers, is part of the BERT model family. This tokenizer is associated with the `bert-base-uncased` model, which has been widely used for tasks such as sentiment analysis, question answering, and text classification. The tokenizer ensures all text is converted to lowercase and accents are removed, reducing vocabulary size and improving generalization. Example:
# 
# - Input: "I Love Transformers."
# - Subword Splits: `['i', 'love', 'trans', '##formers', '.']`
# 
# You can learn more about the tokenizer and model on [Hugging Face's bert-base-uncased page](https://huggingface.co/bert-base-uncased).
# 
# #### Using the `AutoTokenizer` Class
# The `AutoTokenizer` class in the Hugging Face Transformers library provides a seamless way to load tokenizers for various pretrained models. It automatically selects the correct tokenizer configuration based on the model name.
# 

# In[134]:


from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# This initializes a tokenizer tailored for the `bert-base-uncased` model. You can refer to the [Hugging Face AutoTokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer) for more details. You can find the definition of the `from_pretrained()` method [here](https://huggingface.co/docs/transformers/v4.48.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained).
# 
# Instead of tokenizing the entire dataset, we will test the tokenizer on a few sample reviews directly from the dataset:
# 

# In[135]:


# Take sample inputs from the dataset
sample_texts = train_data['review'].sample(3, random_state=42).tolist()

# Tokenize sample inputs
tokenized_samples = tokenizer(sample_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")


# In[136]:


print(tokenized_samples)


# Explanation of parameters:
#  - `truncation=True`: Truncates text longer than the specified max_length.
#  - `padding=True`: Pads shorter sequences to match max_length.
#  - `max_length=128`: Specifies the maximum length of the sequences.
#  - `return_tensors="pt"`: Returns PyTorch tensors as the output format.
#  
# For more details about truncation and padding, refer to the [Hugging Face Padding and Truncation Documentation](https://huggingface.co/docs/transformers/pad_truncation). This step ensures that the tokenizer works as expected and provides insight into its behavior. Next, we will use the tokenizer within the class definition to process the dataset.
# 
# ---
# 

# # Implement a DataLoader in PyTorch<a id="implement-a-dataloader-in-pytorch"></a>
# 
# 
# In this section, you will implement a custom dataset class and use it to create a DataLoader in PyTorch for feeding data into the model during training. PyTorch simplifies this process by providing the `Dataset` and `DataLoader` classes, which handle batching, shuffling, and preprocessing, allowing you to focus on the model architecture and training.
# 
# To start, we will create a custom dataset class for the IMDB dataset, which will process and return tokenized inputs along with their corresponding labels. This class will use a tokenizer to preprocess the raw text data.
# 
# ### 1. Define a Custom Dataset Class
# 
# The custom dataset class will inherit from `torch.utils.data.Dataset` and include the following features:
# 
# 1. **Initialization (`__init__`)**:
#    - Accepts raw text and label data, along with a tokenizer and a maximum sequence length.
#    - The tokenizer is used to preprocess the text data into tokenized inputs.
#    - The maximum sequence length ensures that all tokenized inputs are of uniform length.
# 
# 2. **Length (`__len__`)**:
#    - Returns the total number of data samples in the dataset.
# 
# 3. **Item Retrieval (`__getitem__`)**:
#    - Retrieves a single data point by index.
#    - Preprocesses the text using the tokenizer to create tokenized input IDs.
#    - Returns the tokenized input IDs and the corresponding label for the given index.
# 
# You can refer to [this](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) tutorial on the Pytorch website for more details.
# 

# In[137]:


import torch
from torch.utils.data import Dataset
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# **NOTE ABOUT GPU USAGE**
# 
# The workspace provides you with access to a GPU which is necessary for training a transformer model due to the efficiency provided by GPUs on the large amount of computations that are required. To ensure judicious usage of limited resources, please usage the GPU only when you are training the model. 
# 
# ---
# We will keep the maximum length of input to 128 tokens.

# In[138]:


from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    """
    A custom PyTorch Dataset for the IMDB dataset.

    This class preprocesses text data using a tokenizer and returns tokenized inputs
    along with their corresponding labels for sentiment analysis.

    Attributes:
        data (pd.DataFrame): A DataFrame containing text and label columns.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for preprocessing text.
        max_length (int): Maximum length for tokenized sequences.
    """
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): A DataFrame with columns `review` (text) and `label` (target).
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to preprocess the text.
            max_length (int, optional): Maximum token sequence length. Defaults to 128.
        """
        #TODO: Write code to initialize the dataset
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        #TODO: Write code to return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data point by index and preprocess it.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            torch.Tensor: Tokenized input IDs for the text.
            int: Label corresponding to the text.
        """
        #TODO: Write code for the __getitem__() function
        review = str(self.data.iloc[idx]['review'])
        label = int(self.data.iloc[idx]['label'])

        # Tokenize the review
        encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Squeeze removes the batch dimension (1, max_length) → (max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return input_ids, attention_mask, int(label)


# ### 2. Initialize the Dataset
# 
# Once the `IMDBDataset` class is defined, we can initialize it directly with the training and validation DataFrames.

# In[139]:


# Initialize the datasets
train_dataset = IMDBDataset(train_data, tokenizer)
val_dataset = IMDBDataset(val_data, tokenizer)
test_dataset = IMDBDataset(test_df, tokenizer)


# ### 3. Create a DataLoader
# 
# The `DataLoader` class in PyTorch helps manage batches of data during training. We will use it to create training and validation data loaders.

# In[140]:


from torch.utils.data import DataLoader

# Define batch size
BATCH_SIZE = 32

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 
# You may have seen the use of `RandomShuffler` along with `DataLoader`. That approach is used when you want to train a model over a dataset randomly for a specified number of steps. In this model, we want to use the epoch approach.
# 
# An **epoch** is one complete pass through the entire dataset. When using the `DataLoader` in this setup, it ensures that every data point in the dataset is used exactly once during a single epoch. This approach is helpful for training models in a structured manner, ensuring that the model sees all the training examples and learns from them in each epoch before moving to the next one.
# 
# By setting `shuffle=True` for the `train_loader`, the data points are randomly shuffled at the start of each epoch, improving the generalization of the model. You will see this in action later in the code for training the model.
# 
# ---
# Given below are some assert statements to check your custom dataset and data loader definitions.

# In[141]:


input_ids, attention_mask, label = train_dataset[0]
assert isinstance(input_ids, torch.Tensor), "Input IDs should be a torch.Tensor!"
assert isinstance(label, (int, np.integer)), "Label should be an integer or int-like!"
assert input_ids.shape[0] == train_dataset.max_length, "Input IDs tensor has incorrect length!"
print("--> Dataset test passed!")


# In[142]:


assert len(train_dataset) == 22500, "Train dataset length mismatch!"
assert len(val_dataset) == 2500, "Validation dataset length mismatch!"
assert len(test_dataset) == 25000, "Test dataset length mismatch!"

import numpy as np

# Check the first item in the train dataset
input_ids, attention_mask, label = train_dataset[0]
assert isinstance(input_ids, torch.Tensor), "Input IDs should be a torch.Tensor!"
assert isinstance(label, (int, np.integer)), "Label should be an integer or int-like!"

# Ensure the input IDs tensor has the correct shape
assert input_ids.shape[0] == train_dataset.max_length, "Input IDs tensor has incorrect length!"


# ## <a id="customize-the-transformer-architecture"></a>Customize the Transformer Architecture
# 
# In this section, you will customize the transformer architecture to suit the task of binary classification. You may have used a similar architecture in the past for generation tasks. But you will need to make a few tweaks specifically in the `DemoGPT` class to adapt it for the binary classification.
# 
# ### 1. Config Dictionary
# Your config dictionary bundles all hyperparameters and model settings in one place. Below is the config that we will use in our model:

# In[143]:


config = {
    "vocabulary_size": tokenizer.vocab_size,  # e.g., ~30522 for bert-base-uncased
    "num_classes": 2,                         # binary classification (pos/neg)
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,  # 4 heads * 32 = 128 -> matches d_embed
    "dropout_rate": 0.1,
    "use_bias": True
}


# Key Config Parameters:
# - `vocabulary_size`: The total number of tokens in your vocabulary.
# - `num_classes`: The number of classes for the classification head (2 = binary).
# - `d_embed`: Dimensionality of embeddings (and hidden layers).
# - `context_size`: Maximum sequence length for each input.
# - `layers_num`: Number of stacked transformer blocks.
# - `heads_num`: Number of attention heads in multi-head attention.
# - `head_size`: Dimension of each attention head (must satisfy heads_num * head_size = d_embed).
# - `dropout_rate`: Probability of dropping units during training to reduce overfitting.
# - `use_bias`: Whether linear layers should have bias terms.
# 

# 
# ### 2. Class Definitions
# 
# Below are the class definitions you will work with. These classes form the core components of the transformer model. You may have seen these before, with the exception of the `DemoGPT` class which will need to be customized.
# 
# 
# #### AttentionHead

# In[144]:


import torch.nn as nn
import math

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)  # (B, T, head_size)
        K = self.K_weights(input)  # (B, T, head_size)
        V = self.V_weights(input)  # (B, T, head_size)

        # Q @ K^T => (B, T, T)
        attention_scores = Q @ K.transpose(1, 2)

        # Casual Mask
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
            float('-inf')
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V


# Here we use a dummy input aligned with our config:
# 
# - Batch size = `BATCH_SIZE` (32)
# - Sequence length = `config["context_size"]` (128)
# - Embedding dimension = `config["d_embed"]` (128)
# 

# In[145]:


# Instantiate the AttentionHead
attention_head = AttentionHead(config).to(device)

# Create a dummy input of shape (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)

# Forward pass
attention_output = attention_head(dummy_input)
print("AttentionHead output shape:", attention_output.shape)


# Expected shape:
# 
# >`(B,T,head_size)=(32,128,32)`
# 
# #### MultiHeadAttention
# 

# In[146]:


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)  # (B, T, heads_num * head_size)
        x = self.linear(x)                   # (B, T, d_embed)
        x = self.dropout(x)
        return x


# In[147]:


# Instantiate MultiHeadAttention
multi_head_attention = MultiHeadAttention(config).to(device)

# Same dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)

# Forward pass
mha_output = multi_head_attention(dummy_input)
print("MultiHeadAttention output shape:", mha_output.shape)


# Expected shape:
# 
# >`(B,T,d_embed)=(32,128,128)`
# 
# #### FeedForward
# 

# In[148]:


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)


# In[149]:


# Instantiate FeedForward
feed_forward = FeedForward(config).to(device)

# Dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)

# Forward pass
ff_output = feed_forward(dummy_input)
print("FeedForward output shape:", ff_output.shape)


# Expected shape:
# 
# >`(B,T,d_embed)=(32,128,128)`
# 
# #### Block
# 

# In[150]:


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input):
        x = input
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# In[151]:


# Instantiate a single Block
block = Block(config).to(device)

# Dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)

# Forward pass
block_output = block(dummy_input)
print("Block output shape:", block_output.shape)


# Expected shape:
# 
# >`(B,T,d_embed)=(32,128,128)`
# 
# #### DemoGPT
# Below is the starter code for the `DemoGPT` class, which implements the core of a transformer model tailored for a binary classification task. This implementation builds on the foundation of a transformer architecture and includes the necessary modifications to adapt it for classification. 
# 
# ### Key Changes for Binary Classification
# 
# To adapt the transformer for classification, the following changes are required:
# 
# 1. **Add a Classification-Specific Output Layer**:
#    - The model needs a linear layer to map the final pooled embeddings to the number of classes. 
#    - The classification head is implemented using [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) with:
#      - `in_features` set to `d_embed` (the embedding dimension).
#      - `out_features` set to `num_classes` (the number of classes, 2 for binary classification).
#      - `bias` set to `False` (optional; bias can be excluded to slightly simplify computations).
# 
# 
# 2. **Implement a Pooling Mechanism**:
# 
#    - Transformers output embeddings for each token in the input sequence. For classification, these token-level embeddings need to be condensed into a single vector.
# 
#    - A **mean pooling operation** is applied using [`torch.mean`](https://pytorch.org/docs/stable/generated/torch.mean.html) across the time dimension (`dim=1`) to aggregate token-level embeddings into a single representation vector.

# In[152]:


class DemoGPT(nn.Module):
    def __init__(self, config):
        """
        Initialize the DemoGPT class with configuration parameters.

        Args:
        - config (dict): Configuration dictionary with the following keys:
            - "vocabulary_size": Size of the vocabulary.
            - "d_embed": Dimensionality of the embedding vectors.
            - "context_size": Maximum sequence length (context size).
            - "layers_num": Number of transformer layers.
            - "num_classes": Number of output classes (2 for binary classification).
        """
        super().__init__()
        # Token embedding layer: Maps token indices to embedding vectors.
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        
        # Positional embedding layer: Adds positional information to the embeddings.
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        
        # Transformer layers: Stacked sequence of transformer blocks.
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        
        # Layer normalization: Applied to stabilize training.
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        
        # TODO: Implement classification output layer - Maps pooled embeddings to class logits.
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=config["use_bias"])

    def forward(self, token_ids):
        """
        Forward pass of the model.

        Args:
        - token_ids (torch.Tensor): Input token indices of shape (B, T), 
                                    where B is the batch size, and T is the sequence length.
        
        Returns:
        - logits (torch.Tensor): Output logits of shape (B, num_classes).
        """
        batch_size, tokens_num = token_ids.shape

        # Step 1: Create embeddings for tokens and their positions
        x = self.token_embedding_layer(token_ids)  # Shape: (B, T, d_embed)
        positions = torch.arange(tokens_num, device=token_ids.device)  # Shape: (T,)
        pos_embed = self.positional_embedding_layer(positions)  # Shape: (T, d_embed)
        x = x + pos_embed.unsqueeze(0)  # Add positional embeddings to token embeddings
        
        # Step 2: Pass embeddings through transformer layers
        x = self.layers(x)  # Shape: (B, T, d_embed)
        x = self.layer_norm(x)  # Normalize across the feature dimension
        
        # Step 3: TODO: Apply mean pooling across the time dimension  # Shape: (B, d_embed)
        pooled_output = torch.mean(x, dim=1)  # (B, d_embed)
        
        # Step 4: TODO: Generate logits for classification  # Shape: (B, num_classes)
        logits = self.classifier(pooled_output)  # (B, num_classes)
        
        return logits


# In[153]:


# Instantiate the model
demo_gpt = DemoGPT(config).to(device)

# Suppose we have a batch of size 32, each with a sequence length of 128
dummy_token_ids = torch.randint(
    0, config["vocabulary_size"], 
    (BATCH_SIZE, config["context_size"])
).to(device)

# Forward pass
logits = demo_gpt(dummy_token_ids)

print("DemoGPT output shape:", logits.shape)
print("Logits sample:\n", logits[:2])  # Print first two examples' logits


# Expected shape:
# 
# > `(B,num_classes)=(32,2)`
# 

# In[154]:


# Assertions
assert logits.size(1) == config["num_classes"], "Number of output classes mismatch!"
assert logits.size(0) == BATCH_SIZE, "Batch size mismatch!"
print("--> Model output verified!")


# In[155]:


# Assert that the number of logits matches the number of classes
assert logits.size(1) == config["num_classes"], (
    f"Expected number of classes {config['num_classes']}, "
    f"but got {logits.size(1)}"
)

# Assert that the batch size of the output matches the input batch size
assert logits.size(0) == BATCH_SIZE, (
    f"Expected batch size {BATCH_SIZE}, "
    f"but got {logits.size(0)}"
)


# ## Implement Accuracy Calculation Method <a name="implement-accuracy-calculation-method"></a>
# 
# In this section, you will learn how to calculate the validation accuracy for the transformer model on the IMDB dataset. Validation accuracy provides a performance metric that helps assess how well the model generalizes to unseen data during training.
# 
# ### 1. Overview
# The function to calculate validation accuracy will:
# 
# - Evaluate the model on the validation dataset.
# - Generate predictions for each batch.
# - Compare predictions with the true labels.
# - Compute the percentage of correctly classified examples.
# 
# ### 2.  Key Points
# - **Evaluation Mode**: Calling `model.eval()` ensures that dropout and other training-specific layers are disabled during evaluation.
# - **No Gradients**: The `torch.no_grad()` context disables gradient computation, reducing memory usage and speeding up validation.
# - **Predictions**: `torch.argmax(logits, dim=1)` retrieves the index of the highest logit for each sample, which corresponds to the predicted class label.
# - **Accuracy Calculation**: The function computes the fraction of correct predictions out of the total number of samples, then multiplies by 100 to express it as a percentage.
# 
# After calculating the validation accuracy, incorporate this function into your training loop. Typically, you would call calculate_accuracy at the end of each epoch or after a specific number of training steps. Monitoring validation accuracy over time helps you track performance gains and identify potential overfitting or underfitting issues.

# In[156]:


def calculate_accuracy(model, data_loader, device):
    """
    Calculate the accuracy of the model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained transformer model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model (e.g., 'cuda' or 'cpu').

    Returns:
        float: Validation accuracy as a percentage.
    """
    
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Unpack the batch
            input_ids, attention_mask, labels = batch

            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass (ignore attention_mask if model doesn’t use it)
            logits = model(input_ids)

            # Get predictions
            predictions = torch.argmax(logits, dim=1)

            # Count correct predictions
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Compute accuracy
    accuracy = (total_correct / total_samples) * 100
    return accuracy


# In[157]:


model = DemoGPT(config).to(device)


# In[158]:


validation_accuracy = calculate_accuracy(model, val_loader, device)
print(f"Validation Accuracy: {validation_accuracy:.2f}%")


# As you can see, the validation accuracy is close to 50%. This is expected as the model's parameters have been initialized randomly. 

# ## Train the Model <a name="train-the-model"></a>
# 
# In this section, we will define the training loop for the transformer-based model designed for sentiment analysis. The training loop is crucial for optimizing the model's weights and biases to minimize the loss function and improve classification performance.
# 
# ### Training Loop
# 
# The training loop will involve the following steps:
# 
# 1. **Iterate through epochs**: Repeat the training process for a predefined number of epochs.
# 2. **Load batches of data**: Use the `DataLoader` to retrieve batches of input IDs and labels.
# 3. **Forward pass**: Compute the logits by passing the input IDs through the model.
# 4. **Compute loss**: Use cross-entropy loss as the criterion.
# 5. **Backward pass and optimization**: Backpropagate the loss and update the model parameters using the optimizer.
# 6. **Validation**: Calculate the validation accuracy after each epoch.
# 

# In[159]:


import torch.optim as optim

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
EPOCHS = 5

# Initialize model, loss, and optimizer
model = DemoGPT(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        # Unpack batch correctly (3 elements per batch)
        input_ids, attention_mask, labels = batch

        # Move to device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # TODO: Implement forward pass
        logits = model(input_ids)

        # TODO: Calculate loss
        loss = criterion(logits, labels)

        # TODO: Set gradients to zero
        optimizer.zero_grad()

        # TODO: Backward pass
        loss.backward()

        # TODO: Step the optimizer
        optimizer.step()

        running_loss += loss.item()

        # Log training progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/100:.4f}")
            running_loss = 0.0
   
    # Evaluate validation accuracy
    val_accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")


# #### Notes
# 
# - The `evaluate_accuracy` function calculates the model's accuracy on the validation dataset. Ensure this function is defined and works as expected.
# - The training progress is logged every 100 steps to monitor performance.
# - After each epoch, the validation accuracy is printed to ensure the model generalizes well to unseen data.
# 

# ## Test the Model <a name="test-the-model"></a>
# 
# In this section, you will evaluate the performance of your trained transformer model on the test dataset. Testing the model involves loading the test dataset, passing it through the model, and calculating the accuracy.

# In[160]:


# TODO: Calculate the accuracy of the model over the test set using the calculate_accuracy() function

# Evaluate model performance on the test dataset
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 
# With the accuracy calculated, you can verify if the model meets the project goal of achieving greater than 75% accuracy on the test dataset. 
# 
# Try training your model for more than 3 epochs! Try increasing the size of the embedding used in the model. Or try increasing the number of blocks or layers in the model. You may be able to improve the accuracy of your model further!
# 

# ## Conclusion <a name="conclusion"></a>

#  TODO: 
#  1. Summarize the project results
#  2. List key takeaways

# In[162]:


# ## Conclusion

# 1. Summarize the project results
print("""
In this project, we fine-tuned a transformer-based model (DemoGPT) for binary sentiment classification
on the IMDB dataset. Using PyTorch and Hugging Face's tokenizer, we trained and validated our model
to distinguish between positive and negative reviews.

After training for 5 epochs, the model achieved approximately {:.2f}% accuracy on the validation set
and {:.2f}% accuracy on the test set.
""".format(val_accuracy, test_accuracy))

# 2. List key takeaways
print("""
Key Takeaways:
1. Transformers can effectively model sentiment when paired with token-level attention mechanisms.
2. Tokenization with 'bert-base-uncased' ensures efficient subword representation of text data.
3. Model performance depends heavily on hyperparameters like embedding size, number of layers, and epochs.
4. Validation accuracy monitoring prevents overfitting and guides training optimization.
5. The modular PyTorch Dataset and DataLoader APIs simplify preprocessing and batching for NLP tasks.
""")

