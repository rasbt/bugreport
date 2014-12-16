Sebastian Raschka  
Dec 16, 2014  
Python: 3.4.2   
scikit-learn: 0.15.2   
System info:  GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux

Note: gc.isenabled() evaluates to `True`

The discussion in scikit-learn's GitHub repository can be found here:
[https://github.com/scikit-learn/scikit-learn/issues/3973](https://github.com/scikit-learn/scikit-learn/issues/3973)

This repository contains the code to reproduce this issue. The script can be found in [./code/naive_bayes_scripts/bernoulli_gridsearch_1.py](./code/naive_bayes_scripts/bernoulli_gridsearch_1.py).


## scikit's GridSearch and Python in general are not freeing memory


I made some weird observations that my GridSearches keep failing after a couple of hours and I initially couldn't figure out why. I monitored the memory usage then over time and saw that it it started with a few gigabytes (~6 Gb) and kept increasing until it crashed the node when it reached the max. 128 Gb the hardware can take. 
I was experimenting with random forests for classification of a large number of text documents. For simplicity -- to figure out what's going on -- I went back to naive Bayes.

The versions I am using are 

- Python 3.4.2 
- scikit-learn 0.15.2

I found some related discussion on the scikit-issue list on GitHub about this topic: https://github.com/scikit-learn/scikit-learn/issues/565 and
https://github.com/scikit-learn/scikit-learn/pull/770

And it sounds like it was already successfully addressed!

So, the relevant code that I am using is

    grid_search = GridSearchCV(pipeline, 
                               parameters, 
                               n_jobs=1, # 
                               cv=5, 
                               scoring='roc_auc',
                               verbose=2,
                               pre_dispatch='2*n_jobs',
                               refit=False)  # tried both True and False
    
    grid_search.fit(X_train, y_train)  
    print('Best score: {0}'.format(grid_search.best_score_))  
    print('Best parameters set:') 


Just out of curiosity, I later decided to do the grid search the quick & dirty way via nested for loop

    for p1 in parameterset1:
        for p2 in parameterset2:
            ...
                pipeline = Pipeline([
                            ('vec', CountVectorizer(
                                       binary=True,
                                       tokenizer=params_dict[i][0][0],
                                       max_df=params_dict[i][0][1],
                                       max_features=params_dict[i][0][2],
                                       stop_words=params_dict[i][0][3],
                                       ngram_range=params_dict[i][0][4],)),
                             ('tfidf', TfidfTransformer(
                                          norm=params_dict[i][0][5],
                                          use_idf=params_dict[i][0][6],
                                          sublinear_tf=params_dict[i][0][7],)),
                             ('clf', MultinomialNB())])

                scores = cross_validation.cross_val_score(
                                            estimator=pipeline,
                                            X=X_train, 
                                            y=y_train, 
                                            cv=5, 
                                            scoring='roc_auc',
                                            n_jobs=1)
 
               params_dict[i][1] = '%s,%0.4f,%0.4f' % (params_dict[i][1], scores.mean(), scores.std())
               sys.stdout.write(params_dict[i][1] + '\n')
 
So far so good. The grid search runs and writes the results to stdout. However, after some time it exceeds the memory cap of 128 Gb again. Same problem as with the GridSearch in scikit. After some experimentation, I finally found out that 

    gc.collect()
    len(gc.get_objects()) # particularly this part!

in the for loop solves the problem and the memory usage stays constantly at 6.5 Gb over the run time of ~10 hours.

Eventually, I got it to work with the above fix, however, I am curious to hear your ideas about what might be causing this issue and your tips & suggestions!


