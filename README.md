# book-calibre

I have a pretty large ebook library, mostly fiction in french english or spanish. The idea is to extract all the words of the books (for the moment only in one specific language), and perform a clustering based on the similarity indexes of books, so as to invent custom made tags to regroup books.

We will try to do two things:

**1) Document clustering**
Try to create clusters of similar books, so as to create some kind of recommendation system based on what I already read.
As such we will do (based on Wikipedia for document clustering):
1. Tokenization
2. Stemming and Lemmatization
3. Removing stop words and punctuation
4. Computing term frequencies or tf-idf 
5. Clustering
6. Evaluation and visualization

The second part of the project will be to do
**2) Automatic text suammary**
