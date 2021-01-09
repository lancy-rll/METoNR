# METoNR:A Meta explanation triplet oriented news recommendation model
### Files and folders

* trainsformer folder is an implementation of a transformer layer.
* loadDataset.py loaded neighbors of users and news,train dataset,test dataset and news clicked sequence.
* dataprocess.py constructs a heterogeneous graph, and the generated related files are stored in data/,data/edge/,data/neigb/,topicExtrac.py generates the topic of the news and constructs the edge of the news and the topic.
* word vector is obtained using bert's pre-training model(multi_cased_L-12_H-768_A-12)(https://github.com/google-research/bert).

### Environment:
* python == 3.7.6
* tensorflow==2.3.1
* keras==2.2.4
### Running the code
```
$ python main.py
```
### Dataset
* Due to github restrictions, I put the dataset in the link below