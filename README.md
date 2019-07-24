# AEM

This repo attempts to build the 'Amsterdam Embedding Model' (AEM): A news domain specific word embedding model trained on Dutch journalistic content.

---
This repo contains the following elements:

- Training of word embeddings models:
    - We use Word2Vec to train sets of models on smaller and larger samples with different parameter settings
- Evaluation of these models:
	- Intrinsic evaluation (i.e., syntatic and semantic accuracy of the models), using the following task: [evaluating dutch embeddings](https://github.com/clips/dutchembeddings)
	- Extrinsic evaluation (i.e., performance of the models in downstream tasks)

In doing so, we compare the here-trained models with pre-trained word embedding models on Dutch corpora (i.e., the COW model and a FastText model trained on Wikipedia data, available here: https://github.com/clips/dutchembeddings).

---
## Python scripts:

#### run_classifier.py

This script will execute text classification task, and returns accuracy scores for classification with SGD and ExtraTrees with or without embeddings.

Example usage:

```
python3 run_classifier.py  --word_embedding_path ../folder_with_embedding_models/
--word_embedding_sample_size large  --type_vectorizer Tfidf
--data_path data/dataset_vermeer.pkl --output output/output

```

#### run_intrinsic_evaluation.py
This script returns accuracy scores for instrinc evaluation tasks.

Example usage:

```
python3 run_intrinsic_evaluation.py  --word_embedding_path ../folder_with_embedding_models/ --word_embedding_sample_size large  --output output/output --path_to_evaluation_data ../model_evaluation/analogies/question-words.txt
```
## Directories:

- `lib/`: Modules used in python scripts: Classification
- `output/`: Default output directory
- `helpers/`: small scripts to get info on training samples
-`model_training/`: here you will find code used to train the models.

`make_tmpfileuniekezinnen.py` can be used to extract sentences from articles INCA database. Duplicate sentences will be removed right away.
`trainmodel_fromfile_iterator_modelnumber.py` takes a .txt file with sentences as input and trains word2vec models with different parameter settings.


## Data:

The current study tests the quality of classifiers w/wo embedding vectorizers on the following data:

Vermeer, S.: [Dutch News Classifier](https://figshare.com/articles/A_supervised_machine_learning_method_to_classify_Dutch-language_news_items/7314896/1) --> This datasets classifies Dutch news in four topics

Buscher, Vliegenthart & De Vrees: [Policy Issues Classifier](https://www.google.com/search?q=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&oq=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&aqs=chrome..69i57.688j0j7&sourceid=chrome&ie=UTF-8) --> This paper tests a classifer of 18 topics on Dutch news

We thank the authors of these papers for sharing their data. If there are any issues with the way we handle the data or in case suggestions arise, please contact us.

## Vectorizers:

This projects uses the [embedding vectorizer](https://github.com/ccs-amsterdam/embeddingvectorizer) (credits for Wouter van Atteveld).
