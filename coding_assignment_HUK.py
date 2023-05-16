import pandas as pd
import numpy as np
import warnings
import re
from tqdm import tqdm
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
tqdm.pandas()

def keep_alpha_num(x):
    s = re.sub(r'[.,"\'-_?:!#;]', '', x)
    s = re.sub(r"[\([{})\]]", "", s)
    s = ' '.join(s.split())
    return s


file_path = 'data/Coding_Challenge_NLP/training.csv'

df = pd.read_csv(file_path, header=0, names=['Source_Id', 'Source', 'Sentiment', 'Feedback'])
df.dropna(inplace=True)
# print(df.head())

df['clean_feedback'] = df['Feedback'].apply(lambda x: keep_alpha_num(x))
df['len_feedback'] = df['clean_feedback'].apply(lambda x: len(str(x)))
df = df[df['len_feedback']>10]
df.drop_duplicates(subset=['clean_feedback'], inplace=True)
df.reset_index(drop=True, inplace=True)
from langdetect import detect

# df['lang'] = df['Feedback'].progress_apply(lambda x: tqdm(detect(x)))


# import multiprocessing as mp
# pool = mp.Pool(mp.cpu_count())
# x= df['clean_feedback'].tolist()
# results = pool.map(detect, x)
# pool.close()
# df['lang'] = results

X = df['Feedback'].tolist()[:100]
y = df['Sentiment'].tolist()[:100]

# Let us get the embeddings from the model
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('distiluse-base-multilingual-cased')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# embeddings = embedder.encode(['Hello World', 'Hallo Welt', 'Hola mundo'])
embeddings = tqdm(embedder.encode(X_train))

























# Model Training Phase
# import evaluate
# accuracy = evaluate.load("accuracy")
# id2label = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
#
# label2id = {"Negative": 0, "Positive": 1, "Neutral": 2, "Irrelevant": 3}
#
# from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
# def preprocess_function(examples):
#     return tokenizer(examples, truncation=True)
#
# # df['Tokenized'] = pool.map(preprocess_function, str(x))
# # df['Tokenized'] = df['clean_feedback'].apply(lambda x: preprocess_function(str(x)))
#
# # preprocess_function("This  is love")
#
# model = AutoModelForSequenceClassification.from_pretrained(
#     'xlm-roberta-base', num_labels=4, id2label=id2label, label2id=label2id
# )



























# count = 20000
# for _ in y:
#     count+=1
#     detect(_)
#     print(count)



