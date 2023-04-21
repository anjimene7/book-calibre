import textract, os
import numpy as np
import pandas as pd
import re
import pickle
import iso639
import argparse
import logging
from collections import Counter, OrderedDict
from xml.etree import ElementTree as ET
from typing import List, Dict, Tuple
import multiprocessing as mp
import itertools
from sklearn.feature_extraction.text import TfidfTransformer

class Book():
    def __init__(self, path: str, method='lemmatization', stopwords='spacy'):
        """
        Class constructor
        :param path: Absolute path to the epub file of the book
        """
        self.path = os.path.dirname(path)
        self.book_filename = os.path.basename(path)
        self.author = os.path.basename(os.path.dirname(self.path))
        self.language, self.subject, self.title = self._extract_metadata()
        self.text, self.final, self.length_book_unique, self.length_book_full = self._extract_words(method=method, stopwords=stopwords)

    def __str__(self):
        return f'Filename: {self.book_filename}, Author: {self.author}, Path: {self.path}, ' \
               f'Language: {self.language}, Subject: {self.subject}, Title: {self.title}'

    def _extract_metadata(self) -> Tuple[str, str, str]:
        """
        This method will extract the Calibre metadata from the epub : language, title, subject
        :return metadata: Metadata dictionnary
        """
        metadata: Dict[str, str] = {}
        tree = ET.parse(os.path.join(self.path, 'metadata.opf'))
        for x in ["title", "subject", "language"]:
            field = [x.text for x in tree.findall(".//{http://purl.org/dc/elements/1.1/}" + x)]
            metadata[x] = field if len(field) == 1 else [",".join(field)]
        metadata['language'] = [str.lower(re.findall(r'\w[\w-]+', iso639.to_name(metadata['language'][0]))[0])]
        return metadata['language'], metadata['subject'], metadata['title']

    def _extract_words(self, method: str, stopwords: str) -> Tuple[List[str], OrderedDict, int, int]:
        """
        Open ebook, parse text, extract words (BoW)
        :param method:
        :param stopwords:
        :return:
        """
        final = dict()
        text = []

        # Extract text, find words, put in an ordered Dict
        try:
            text = textract.process(os.path.join(self.path, self.book_filename))
        except:
            print("Problem with book!: " + self.author + ", " + self.path)
        else:
            text = str(text, "utf-8")
            text = text.replace(u'\xad', '')  # see below
            # These are characters that mark places where a word could be split when fitting lines to a page.
            # The idea is that the soft hyphen is invisible if the word doesn't need to be split, but printed the same as a U+2010 normal hyphen if it does.
            text = text.replace(u'\xa0', u' ')
            text = re.findall(r'\w[\w-]+', text)
            text = [x.lower() for x in text]
            ###############
            if method == 'stemming':
                from nltk.stem import SnowballStemmer
                stemmer = SnowballStemmer(self.language)
                text = [stemmer.stem(word) for word in text]
            elif method == 'lemmatization':
                import spacy
                #nlp = spacy.load(iso639.to_iso639_1(self.language[0]))
                # python -m spacy download fr_dep_news_trf
                nlp = spacy.load("fr_dep_news_trf")
                p_text = " ".join(text)
                nlp.max_length = len(p_text) + 50
                doc = nlp(p_text, disable=['parser', 'tagger', 'ner'])
                text = [token.lemma_ for token in doc]
            elif method == 'none':
                pass
            else:
                raise ValueError("method must be either 'stemming', 'lemmatization' or 'none' !")
            ###############
            customized_stop_words = ('xhtml', '-', 'qu')
            if stopwords == 'nltk':
                import nltk
                from nltk.corpus import stopwords
                nltk.data.path.append("/utils")
                stopwords = set(stopwords.words(self.language[0]))
                stopwords.update(customized_stop_words)

            elif stopwords == 'spacy':
                import spacy
                stopwords = getattr(spacy.lang,
                                    iso639.to_iso639_1(self.language[0])).stop_words.STOP_WORDS
                stopwords.update(customized_stop_words)

            else:
                stopwords = set(customized_stop_words)

            text = [x for x in text if x not in stopwords]
            # keep only words
            masking = [False if re.match(r'^[^a-z]*[0-9]+[^a-z]*$', x) else True for x in text]
            text = [i for (i, v) in zip(text, masking) if v]

            d = dict(Counter(text).items())
            final = OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=True))

            length_book_unique = len(set(text))
            length_book_full = len(text)

        return text, final, length_book_unique, length_book_full

class Application:
    def __init__(self, data_path: str, books_cache_path: str, parallel_pool: int, sample: int = -1, reset_cache: bool = False):
        self.data_path = data_path
        self.books_cache_path = books_cache_path
        self.parallel_pool = parallel_pool
        if reset_cache:
            self._reset_cache()
        self.authors = [x for x in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, x))]
        logging.debug(f"Detected authors: {sorted(self.authors)}")
        if not self.authors:
            logging.warning(f"No books detected")
        else:
            self.files: List[str] = []
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.epub') and os.path.join(root, file) not in self.files:
                        self.files.append(os.path.join(root, file))
            if sample > 0:
                self.files = self.files[0:sample]
                logging.info(f"Selecting only a subset of files (n={len(self.files)})")
            self.books: List[Book] = self._instantiate_books()
            #self._compute_IDF()

    def __str__(self) -> str:
        """
        Returns the representation of the application when the object is printed
        """
        return f"Number of books: {len(self.books)}, Data path: {self.data_path}, Cache path: {self.books_cache_path}"

    def _reset_cache(self) -> None:
        """
        Removes all pickle files from the cache folder
        """
        logging.info(f"Resetting cache: Will remove all pickle files under {self.books_cache_path}")
        for file in os.listdir(self.books_cache_path):
            os.remove(os.path.join(self.books_cache_path, file))

    @classmethod
    def _process_book(cls, file: str, books_cache_path: str):
        pickle_file = "".join([s.lower().replace(' ', '_') for s in os.path.basename(file) if s.isalnum() or s.isspace()]) + '.pickle'
        pickle_file_path = os.path.join(books_cache_path, pickle_file)
        if os.path.exists(pickle_file_path):
            logging.debug(f"Pickle file exists for {file}, will load it from there")
            with open(pickle_file_path, 'rb') as f:
                book = pickle.load(f)
        else:
            book = Book(path=file)
            logging.debug(f"Dumping file into {pickle_file_path}")
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(book, f)
        return book

    def _instantiate_books(self) -> List[Book]:
        """
        This method will instantiate all Books objects for the selected sample. It will load from the cache in case
        it has already been processed before
        """
        logging.debug(f"Will instantiate books with a pool of {self.parallel_pool}")
        if not os.path.isdir(self.books_cache_path):
            logging.info(f"Creating books cache dir at {self.books_cache_path}")
            os.mkdir(self.books_cache_path)

        with mp.Pool(self.parallel_pool) as pool:
            books = pool.starmap(self._process_book, zip(self.files, itertools.repeat(self.books_cache_path)))
        return books

    def _compute_IDF(self):
        # DF is a PD dataframe with rows as documents, and columns as tokens
        df = pd.DataFrame({'words': [x.final for x in self.books]}, index=[x.title[0] for x in self.books])
        df = pd.json_normalize(df['words'])
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(df)
        # df_idf = pd.DataFrame(tfidf_transformer.idf_,index=df.columns, columns=["tf_idf_weights"])
        return tfidf_transformer.idf_

    def _compute_TF(self):
        # DF is a PD dataframe with rows as documents, and columns as tokens
        return df.div(df.sum(axis=1), axis=0)





    def refine_dict(df, lang='french'):
        # 1) Drops comic books (BD), or Audiobooks,
        # 2) drops books based on the language,
        # 3) drops books based on whether it could detect words or not
        # Only keep books for specific language, 'fra' by default since most of the books I have are in French

        ######  1) REMOVE AUDIOBOOKS OR COMIC BOOKS
        rows_to_remove = df['metadata'].index[np.where((df['metadata']['subject'] == 'Audiobook') |
                                                       (df['metadata']['subject'] == 'BD'))[0]]
        for i in df.keys():
            df[i].drop(rows_to_remove, axis=0, inplace=True)
        #######  2) LANGUAGE
        rows_to_remove = df['metadata'].index[np.where(df['metadata']['language'] != lang)[0]]
        for i in df.keys():
            df[i].drop(rows_to_remove, axis=0, inplace=True)
        #######  3) BOOKS WITH NOT ENOUGH WORDS
        # how to know whether a book has not enough words? Most of the books are epub format, words can be easily scrapped,
        # some books are pdf and the textract process does not necesarily extract them.
        # As such we will remove them as a first naive approach. Effectively, this means removing books with length_full of 0
        # up to 900
        rows_to_remove = df['metadata'].index[np.where(df['metadata']['length_full'] < 900)[0]]
        for i in df.keys():
            df[i].drop(rows_to_remove, axis=0, inplace=True)

        #####  4) NOW WE WILL REMOVE THE WORDS THAT ARE NUMBERS (The probably are page numbers, or others. We dont need them)
        # We will for instance remove '112' or '12-13' but not 'paris12', which we will keep stripping the numbers
        cols_to_remove = [x for x in df['words'].columns if re.findall(r'^[^a-z]*[0-9]+[^a-z]*$', x)]
        df['words'].drop(cols_to_remove, axis=1, inplace=True)
        # Now strip numbers from column names
        cols_to_rename = [re.sub(r'[\d]+', '', x) for x in df['words'].columns]
        df['words'].columns = cols_to_rename

        ####  5) REMOVE COLUMNS WITH ONLY NANS IN THE WORDS DF
        cols_to_remove = [x for (x, v) in zip(df['words'].columns, df['words'].isna().all(axis=1)) if v]
        df['words'].drop(cols_to_remove, axis=1, inplace=True)

        ####  6) REMOVE COLUMNS WITH WORDS WITH LEN <= 2 --> NOT REAL WORDS
        cols_to_remove = [x for x in df['words'].columns if len(x) <= 2]
        df['words'].drop(cols_to_remove, axis=1, inplace=True)

        ####  7) FILL NAN values with 0
        df['words'].fillna(0, inplace=True)

        ####  8) MERGE COLUMNS WITH THE SAME NAME --> SINCE THEY HAVE BEEN LEMATIZED
        df['words'] = df['words'].groupby(level=0, axis=1).sum()


def main_process(verbose, method, stopWords, ebookPath, outputPath):
    # MAIN LOOP
    counter = 0
    metadata_dict = dict()
    words_list = []

    metadata = pd.DataFrame(metadata_dict, index=metadata_dict['title'])
    words = pd.DataFrame(words_list, index=metadata_dict['title'])
    final = {'metadata':metadata, 'words':words}
    to_return = {'metadata':metadata, 'words':words}
    
    # NOW CLEAN DATA BEFORE PROCESSING
    refine_dict(final, lang = 'french')
    
    # PROCESS DATA (TFIDF)
    words = compute_TF(final['words'])*compute_IDF(final['words'])
    final['words'] = words
    
    with open(os.path.join(outputPath,'final.pickle'), 'wb') as handle:
        pickle.dump(final, handle, protocol = 4)   

    #os.remove('./runnning')
    print('Process ended')
    return to_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path where data is located", action="store")
    parser.add_argument("-l", "--log_level", help="Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
                        default="INFO", action="store")
    args = parser.parse_args()
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, format=format)

    app = Application(data_path=args.path, books_cache_path='cache', parallel_pool=1, sample=9)
    print('hey')
