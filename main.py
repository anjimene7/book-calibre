import os, sys
import numpy as np
import pandas as pd
import re
import pickle
import iso639
import time
from collections import Counter
from tqdm import tqdm
from pathlib import Path

from process import main_process

os.environ["NLTK_DATA"] = "./utils/"

main_process(True,'stemming','nltk','../../Calibre','output')

with open('final.pickle', 'rb') as handle:
    final = pickle.load(handle)

final['metadata'].head()

final['words'].head()

print('Shape of metadata DF: '+str(final['metadata'].shape))
print('\nShape of words DF: '+str(final['words'].shape))

ax=final['metadata']['length_full'].sort_values().plot(logy=True)
a=np.round(np.linspace(0,final['metadata'].shape[0],15))
ax.xaxis.set_ticks(a)
ax.set_xticklabels(a)
plt.xlabel('Number of books')
plt.ylabel('Length of books')


values = (100*final['metadata']['length_unique']/final['metadata']['length_full']).sort_values(ascending=False)
authors = final['metadata']['author'].loc[values.index].values
subjects = final['metadata']['subject'].loc[values.index].values
df_length = pd.DataFrame(list(zip(authors,values,subjects)), columns=['Authors','Ratio','Subjects'])
df_length = df_length.groupby('Authors').agg({'Ratio':'median', 'Subjects':lambda col: ','.join(col)})

def my_func(x):
    return(Counter(x.split(',')).most_common()[0][0])
subjects = [my_func(x) for x in df_length['Subjects']]

dd = defaultdict(lambda: 'b')
dd['Science-Fiction'] = 'r'
dd['Fantasy'] = 'r'
dd['Littérature'] = 'g'
dd['Historique'] = 'y'
dd['Histoire'] = 'y'
dd['Philosophie'] = 'g'
dd['Roman historique'] = 'y'
dd['Romans historiques'] = 'y'

df_length['Subjects'] = subjects
df_length.reset_index().sort_values('Ratio', ascending=False).plot.bar(x='Authors', y='Ratio',
                                     color= df_length['Subjects'].map(dd))

sf = mpatches.Patch(color='r', label='Science-Fiction, Fantasy')
li = mpatches.Patch(color='g', label='Littérature, Philosophie')
hi = mpatches.Patch(color='y', label='Histoire, Historique')
bl = mpatches.Patch(color='b', label='Autre')
plt.ylabel('Ratio unique words/full length')
_=plt.legend(handles=[sf, li, hi, bl])

list(final['words'].index)

def plot_wordcloud(data):
    from wordcloud import WordCloud
    wordcloud2 = WordCloud(background_color='white',
                           max_words=400,
                           max_font_size=80,
                           width=800, height=400,
                           normalize_plurals=False
                             )
    wordcloud2.generate_from_frequencies(data.to_dict())
    fig = plt.figure(figsize = (40,20))
    plt.imshow(wordcloud2, interpolation='bilinear')
    fig.suptitle(data.name, fontsize=70)

number = int([i for i, s in enumerate(final['words'].index) if 'Candide' in s][0])
data = final['words'].iloc[number]
data.sort_values(ascending=False)

final['words']['homme']

# FICTION
number = int([i for i, s in enumerate(final['words'].index) if 'Candide' in s][0])
data = final['words'].iloc[number]
plot_wordcloud(data)

number = int([i for i, s in enumerate(final['words'].index) if 'Fondation' in s][1])
data = final['words'].iloc[number]
plot_wordcloud(data)

# NON-FICTION
number = int([i for i, s in enumerate(final['words'].index) if 'Sapiens' in s][0])
data = final['words'].iloc[number]
plot_wordcloud(data)

number = int([i for i, s in enumerate(final['words'].index) if 'Mousquetaires' in s][0])
data = final['words'].iloc[number]
plot_wordcloud(data)

number = int([i for i, s in enumerate(final['words'].index) if 'Kilo' in s][0])
data = final['words'].iloc[number]
plot_wordcloud(data)

plot_wordcloud(final['words'].sum(axis=0))

final['words'].mean(axis=0).sort_values(ascending=False)

[x for x,y in Counter(final['words'].columns).items() if y==1]

nr=[i for i, s in enumerate(final['words'].index) if 'Mousquetaires' in s][0]

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist

h = pairwise_distances(final['words'], metric='cosine') #cosine, cityblock, correlation
#h = pdist(final['words'], metric='cosine')

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='aoning', api_key='pXneTE0EeDB2YsrRCEZo')

numbi = len(final['words'].index)
numbi = 500

trace = go.Heatmap(z=h[0:numbi,0:numbi],
                   x=final['words'].index[0:numbi],
                   y=final['words'].index[0:numbi], reversescale = True)


data=[trace]
py.iplot(data, filename='labelled-heatmap')

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

link = hierarchy.linkage(h[0:numbi], method='single')
o1 = hierarchy.leaves_list(link)

mat = h[o1,:]
mat = mat[:, o1[::-1]]
mat = mat[::-1,:]
labs_mat = final['words'].index[o1]
labs_mat = labs_mat[::-1]

trace = go.Heatmap(z=mat[0:numbi,0:numbi],
                   x=labs_mat[0:numbi],
                   y=labs_mat[0:numbi], reversescale = True)


data=[trace]
py.iplot(data, filename='labelled-heatmap')


