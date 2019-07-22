
# Bionexo Clustering


This repository contains the code used to solve the problem of categorization of hospital products(+ 600k samples) of the company Bionexo, proposed in the [V Workshop of Mathematical Solutions for Industrial Problems organized by the Center of Mathematics applied to the Industry (CeMEAI)](http://www.cemeai.icmc.usp.br/component/jem/event/77-v-workshop-de-solucoes-matematicas-para-problemas-industriais).


## Getting Started

### Methodology

We use the techniques of TF-IDF and W2Vec to represent the features of the products in high dimension. Then we use the Support Vector Machine (SVM) technique to classify the products.

### Repository Description

dataset folder contains raw_data.csv with raw data of the products and data_preprocessed.csv contains the records without duplicates.

TF-IDF.ipynb : Contains the code to generate the vectors tf-idf using the description of the products, The vectors are already generated in the file tfidf_vectors.npz.

w2v.ipynb : 


### Installing

We use Python 3
```
pip install sklearn
pip install scipy
pip install nltlk

```

## Authors

**Jorge Poco** , **Elio Rodriguez** and **Joao Pinheiro**

See also the list of [contributors](https://github.com/visual-ds/bionexo_clustering/contributors) who participated in this project.




