## Pandas
- crosstab

## scikit-learn
- all objects in scikit-learn's API share a consistent interface
  - estimators:
    - fit(): takes dataset as param; additionally labels for supervised algos
    - for estimator functions (which estimates a value) like Imputer
    - any hyperparam must be set as a constructor to the imputer object
  - transformers:
    - transform(): takes the dataset as param
      - it returns the transformed dataset; transformation is done by doing some learning on the dataset
    - fit_transform() to do fit() and transform() in one step and sometimes it is optimized to be faster
    - some estimators like imputer can trasnform a dataset
    - we can create custom transformers using sklearn
  - predictors:
    - predict()
    - capable of making predictions
    - generally also has a score() method to score against test data
  - inspection:
    - all the hyperparams of an estimator is available via imputer.strategy
    - all the estimator's learned params are available via imputer.statistics_
  - datasets are represented either as numpy arrays or scipy sparse matrices 
  - uses sensible defaults
  - composition: ex. it's easy to create a pipeline by combining transformers