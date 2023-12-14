import os, openai, torch, clip
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # needed to load the T5 model
from sentence_transformers import SentenceTransformer

openai.api_key_path = os.path.join("./openai_key/key.conf")


def get_embedding_model(embedding_model_type):
    if embedding_model_type == "GUSE_transformer":
        module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/large/versions/2"
    elif embedding_model_type == "GUSE_DAN":
        module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
    elif embedding_model_type == "T5":
        module_url = "https://www.kaggle.com/models/google/sentence-t5/frameworks/TensorFlow2/variations/st5-base/versions/1"
    elif embedding_model_type == "USE_CMLM_Base":
        preprocessor = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3")
        encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")
        return (preprocessor, encoder)
    elif embedding_model_type == "openai_ada2":
        return None
    elif 'clip' in embedding_model_type.lower():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if 'vit' in embedding_model_type.lower():
            model, preprocess = clip.load('ViT-B/32', device)
        elif 'rn50' in embedding_model_type.lower():
            model, preprocess = clip.load('RN50', device)
        else:
            raise ValueError(f"model_name {embedding_model_type} not recognized")
        return model
    else:
        try:
            # default behaviour is to try and load from SentenceTransformer
            return SentenceTransformer(embedding_model_type)
        except Exception as e:
            raise Exception("embedding_model_type not understood")

    models_dir = "./embedding_models"
    os.makedirs(models_dir, exist_ok=True)
    this_embedding_model_dir = os.path.join(f"{models_dir}/{embedding_model_type}")

    if not os.path.exists(f"{this_embedding_model_dir}/saved_model.pb"):
        model = hub.load(module_url)
        tf.saved_model.save(model, this_embedding_model_dir)
        print(f"module {module_url} loaded")

    else:
        model = hub.load(this_embedding_model_dir)
        print(f"module {this_embedding_model_dir} loaded")

    return model


def get_embeddings(sentences, embedding_model, embedding_model_type):
    if embedding_model_type == "T5":
        return embedding_model(sentences)[0].numpy()
    elif embedding_model_type == "USE_CMLM_Base":
        preprocessor, encoder = embedding_model[0], embedding_model[1]
        return encoder(preprocessor(sentences))["default"]
    elif embedding_model_type == "openai_ada2":
        openai_out = openai.Embedding.create(input=sentences, model="text-embedding-ada-002")["data"]
        embeddings = np.asarray([out["embedding"] for out in openai_out])
        return embeddings
    elif embedding_model_type in ["GUSE_transformer", "GUSE_DAN"]:
        return embedding_model(sentences).numpy()
    elif 'clip' in embedding_model_type.lower():
        with torch.no_grad():
            sentences = torch.cat([clip.tokenize(s) for s in sentences])
            return embedding_model.encode_text(sentences).numpy()
    else:
        # default behaviour is to use SentenceTransformer models (e.g. mpnet, etc.) 
        return embedding_model.encode(sentences)


# Helper function to load fasttext vectors
def load_word_vectors(fname, embedding_type):
    
    data = {}
    if embedding_type == 'fasttext':
        try:
            fin = open(fname, encoding="utf-8", newline="\n", errors="ignore")
        except ValueError:
            raise Exception(f"{fname} not found. Localize the .vec containing the embeddings, or download wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
        for line in fin:
            values = line.rstrip().split(" ")
            word = values[0]
            vector = map(float, values[1:])
            vector = np.array([i for i in vector])
            data[word] = vector

    elif embedding_type == 'glove':
        import pandas as pd
        import csv
        try:
            data = pd.read_table(fname, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        except ValueError:
            raise Exception(f"{fname} not found. Localize the .txt containing the embeddings, or download https://nlp.stanford.edu/projects/glove/")
    else:
        raise Exception(f"embedding_type {embedding_type} not understood")

    return data


def get_word_embedding(word, embeddings, embedding_type):
    
    if embedding_type == 'fasttext':
        return embeddings[word]

    elif embedding_type == 'glove':
        return embeddings.loc[word].to_numpy()
    
    else:
        return get_embeddings([word], embeddings, embedding_type)[0]