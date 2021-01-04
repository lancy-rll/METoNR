import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

prefix='F:/newsdataset/data/'
prefixe='F:/newsdataset/data/edge/'
n_keywords=prefix+'news_keys'
news_topic=prefixe+'nt'


n_features=1723
n_topic=50

keylist=[]
with open(n_keywords,'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip('\n')
        keylist.append(line.strip())
tf_vectorizer=CountVectorizer(max_df=0.95,min_df=1)
tf=tf_vectorizer.fit_transform(keylist)

lda=LatentDirichletAllocation(n_topic,max_iter=50,learning_method='batch')
topic=lda.fit_transform(tf)
t_index=np.argmax(topic,axis=1)
idx=0
other_idx=[]
with open(n_keywords,'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip('\n')
        if line=='other':
            other_idx.append(idx)
        idx+=1
with open(news_topic,'w',encoding='utf-8') as f:
    for news_id, topic_idx in enumerate(t_index):
        if news_id not in other_idx:
            f.write(str(news_id+1)+'\t'+str(topic_idx+1)+'\n')

print(topic)
