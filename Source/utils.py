import openml
import tpot2
from tpot2.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
import sklearn.metrics
import sklearn
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
from functools import partial
from estimator_node_gradual import EstimatorNodeGradual

# function to check if any number in a numpy.ndarray is negative
def any_negative(arr):
    return np.any(arr < 0)

# lexicase selection with ignoring the complexity column
def lexicase_selection_no_comp(scores, k, rng=None, n_parents=1,):
    """
    Select the best individual according to Lexicase Selection, *k* times.
    The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :returns: A list of indices of selected individuals.
    """
    rng = np.random.default_rng(rng)
    chosen =[]

    for _ in range(k*n_parents):
        candidates = list(range(len(scores)))
        cases = list(range(len(scores[0]) - 1)) # ignore the last column which is complexity
        rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max(scores[candidates,cases[0]])
            candidates = [x for x in candidates if scores[x, cases[0]] == best_val_for_case]
            cases.pop(0)
        chosen.append(rng.choice(candidates))

    return np.reshape(chosen, (k, n_parents))

# generate scores for lexicase selection to use
def lex_selection_objectives(est,X,y,X_select,y_select,classification):
    # fit model
    est.fit(X,y)

    if classification:
        # classification: wanna maximize the number of correct predictions
        return list(np.float32(est.predict(X_select) == y_select)) + [np.int64(tpot2.objectives.complexity_scorer(est,0,0))]
    else:
        # regression: wanna minimize the distance between predicted and true values
        return list(np.absolute(y_select - est.predict(X_select), dtype=np.float32)) + [np.int64(tpot2.objectives.complexity_scorer(est,0,0))]

# generate scores for tournament selection to use
def aggregated_selection_objectives(est,X,y,X_select,y_select,classification):
    # fit model
    est.fit(X,y)

    if classification:
        # classification: wanna maximize the number of correct predictions
        return [np.mean(est.predict(X_select) == y_select, dtype=np.float32), np.int64(tpot2.objectives.complexity_scorer(est,0,0))]
    else:
        # regression: wanna minimize the distance between them
        return [np.mean(np.absolute(y_select - est.predict(X_select), dtype=np.float32)), np.int64(tpot2.objectives.complexity_scorer(est,0,0))]

# get selection scheme
def get_selection_scheme(scheme, classification):
    if scheme == 'lexicase' and classification == True:
        return lexicase_selection_no_comp
    elif scheme == 'tournament':
        return tpot2.selectors.tournament_selection
    else:
        raise ValueError(f"Unknown selection scheme: {scheme}")

# pipeline search space: selector(optional) -> transformer(optional) -> regressor/classifier(required)
def get_pipeline_space(classification, seed):
    if classification:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_classification","Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
            tpot2.config.get_search_space(["transformers","Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
            tpot2.config.get_search_space("classifiers", random_state=seed, base_node=EstimatorNodeGradual)])
    else:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_regression","Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
            tpot2.config.get_search_space(["transformers","Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
            tpot2.config.get_search_space("regressors", random_state=seed, base_node=EstimatorNodeGradual)])

# get estimator parameters depending on the selection scheme
def get_estimator_params(n_jobs,
                         classification,
                         scheme,
                         split_select,
                         X_train,
                         y_train,
                         seed):
    # split the training data
    print('(train)', 1.0-split_select, '/ (select)', split_select)
    if classification:
        X_learn, X_select, y_learn, y_select = sklearn.model_selection.train_test_split(X_train,
                                                                                        y_train,
                                                                                        train_size=1.0-split_select,
                                                                                        test_size=split_select,
                                                                                        stratify=y_train,
                                                                                        random_state=seed)
    else:
        X_learn, X_select, y_learn, y_select = sklearn.model_selection.train_test_split(X_train,
                                                                                        y_train,
                                                                                        train_size=1.0-split_select,
                                                                                        test_size=split_select,
                                                                                        random_state=seed)
    print('X_learn:',X_learn.shape,'|','y_learn:',y_learn.shape)
    print('X_select:',X_select.shape,'|','y_select:',y_select.shape)

    print('Any negative values in X_learn:', any_negative(X_learn))
    print('Any negative values in X_select:', any_negative(X_select))

    # negative weights for regression tasks and positive weights for classification tasks
    objective_weight = -1.0 if not classification else 1.0

    # get selection objective functions
    if scheme == 'lexicase':
        # create selection objective functions
        objective_scorer = partial(lex_selection_objectives,X=X_learn,y=y_learn,X_select=X_select,y_select=y_select,classification=classification)
        objective_scorer.__name__ = 'selection-objectives'
        # create list of objective names per sample in y_select + complexity
        objective_names = ['obj_'+str(i) for i in range(y_select.shape[0])] + ['complexity']
        objective_weights = [1.0 for _ in range(y_select.shape[0])] + [-1.0]

    elif scheme == 'tournament' or scheme == 'random':
        # create selection objective functions
        objective_scorer = partial(aggregated_selection_objectives,X=X_learn,y=y_learn,X_select=X_select,y_select=y_select,classification=classification)
        objective_scorer.__name__ = 'accuracy-complexity'
        # accuracy + complexity
        objective_names = ['performance', 'complexity']
        objective_weights = [objective_weight, -1.0]

    else:
        raise ValueError(f"Unknown selection scheme: {scheme}")

    return objective_names, {
        # evaluation criteria
        'scorers': [],
        'scorers_weights':[],
        'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed), # not used
        'other_objective_functions': [objective_scorer],
        'other_objective_functions_weights': objective_weights,
        'objective_function_names': objective_names,

        # evolutionary algorithm params
        'population_size' : 100,
        'generations' : 200,
        'n_jobs':n_jobs,
        'survival_selector' :None,
        'parent_selector': get_selection_scheme(scheme, classification),
        'random_state': seed,

        # offspring variation params
        'mutate_probability': 0.7,
        'crossover_probability': 0.0,
        'crossover_then_mutate_probability': 0.3,
        'mutate_then_crossover_probability': 0.0,

        # estimator params
        'memory_limit':0,
        'preprocessing':False,
        'classification' : classification,
        'verbose':1,
        'max_eval_time_mins':5, # 5 min time limit
        'max_time_mins': float("inf"), # run until generations are done

        # pipeline search space
        'search_space': get_pipeline_space(classification, seed)
        }

# get test scores
def score(est, X, y, X_train, y_train, classification):
    # train evovled pipeline on the training data
    est.fit(X_train, y_train)

    # calculate testing performance
    if classification:
        # get classification testing score
        performance = np.float32(sklearn.metrics.get_scorer("accuracy")(est, X, y))
    else:
        # get regression mean absolute error score
        performance = np.float32(sklearn.metrics.get_scorer("neg_mean_absolute_error")(est, X, y))
    return {'testing_performance': np.absolute(performance, dtype=np.float32), 'testing_complexity': np.int64(tpot2.objectives.complexity_scorer(est,0,0))}

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, classification, preprocess=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)

        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

            # needed this to LabelEncode the target variable if it is a classification task only
            if classification:
                le = sklearn.preprocessing.LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test

# get the best pipeline from tpot2 depending on the selection scheme
def get_best_pipeline_results(est, obj_names, scheme, seed, classification):
    # get correct subset of evaluated individuals and drop NaN values
    if scheme == 'lexicase':
        sub = est.evaluated_individuals.dropna(subset=['obj_0'])
        sub['performance'] = est.evaluated_individuals[obj_names[:-1]].mean(axis=1)
    else:
        sub = est.evaluated_individuals.dropna(subset=['performance'])

    # get best performers based on classification or regression
    if classification:
        best_performers = sub[sub['performance'] == sub['performance'].max()]
    else:
        best_performers = sub[sub['performance'] == sub['performance'].min()]

    # filter by the smallest complexity
    best_performers = best_performers[best_performers['complexity'] == best_performers['complexity'].min()]

    # get best performer performance and cast to numpy float32
    best_performer =  best_performers.sample(1, random_state=seed)

    # return performance, complexity, and individual
    return np.float32(best_performer['performance'].values[0]), \
                np.int64(best_performer['complexity'].values[0]), \
                best_performer['Individual'].values[0].export_pipeline()

# execute task with tpot2
def execute_experiment(split_select, scheme, task_id, n_jobs, save_path, seed, classification):
    # generate directory to save results
    save_folder = f"{save_path}/{seed}-{task_id}"
    if os.path.exists(save_folder):
        print('FOLDER ALREADY EXISTS:', save_folder)
        return

    # run experiment
    try:
        print("LOADING DATA")
        X_train, y_train, X_test, y_test = load_task(task_id, preprocess=True, classification=classification)

        # get estimator parameters
        names, est_params = get_estimator_params(n_jobs=n_jobs,classification=classification,scheme=scheme,split_select=split_select,X_train=X_train,y_train=y_train,seed=seed)
        est = tpot2.TPOTEstimator(**est_params)

        start = time.time()
        print("ESTIMATOR FITTING")
        est.fit(X_train, y_train) # x_train, y_train are not used at all by the estimator
        duration = time.time() - start
        print("ESTIMATOR FITTING COMPLETE:", duration / 60 / 60, 'hours')

        print('CREATING FOLDER:', save_folder)
        os.makedirs(save_folder)
        print('SAVING:ESTIMATOR.PKL')

        tmpdf = est.evaluated_individuals
        tmpdf["Individuals_str"] = tmpdf["Individual"].apply(lambda x: str(x))
        tmpdf["Instance_str"] = tmpdf["Instance"].apply(lambda x: str(x))
        tmpdf.drop(columns=["Individual", "Instance"]).to_csv(f"{save_folder}/evaluated_individuals.csv")

        with open(f"{save_folder}/evaluated_individuals.pkl", "wb") as f:
                        pickle.dump(est.evaluated_individuals, f)

        # get best performer performance and cast to numpy float32
        train_performance, complexity, pipeline = get_best_pipeline_results(est, names, scheme, seed, classification)

        print('best pipeline:')
        print(pipeline)
        print('*'*100)

        # save pipeline in a pickle file
        with open(f"{save_folder}/pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        # get test scores and save results
        results = score(pipeline, X_test, y_test, X_train=X_train, y_train=y_train, classification=classification)
        results['training_performance'] = train_performance
        results['training_complexity'] = complexity
        results["task_id"] = task_id
        results["selection"] = scheme
        results["seed"] = seed

        print('RESULTS:', results)

        print('SAVING:SCORES.PKL')
        with open(f"{save_folder}/results.pkl", "wb") as f:
            pickle.dump(results, f)
        return

    # estimator may fail to fit when finding the best pipeline
    # we still have access to the estimated_individuals df
    # so we can still get the best performer mannually and get its results
    except Exception as e:
        duration = time.time() - start
        print("ESTIMATOR FITTING COMPLETE:", duration / 60 / 60, 'hours')

        # get best performer performance and cast to numpy float32
        train_performance, complexity, pipeline = get_best_pipeline_results(est, names, scheme, seed, classification)

        # get test scores and save results
        results = score(pipeline, X_test, y_test, X_train=X_train, y_train=y_train, classification=classification)
        results['training_performance'] = train_performance
        results['training_complexity'] = complexity
        results["task_id"] = task_id
        results["selection"] = scheme
        results["seed"] = seed

        print('RESULTS:', results)

        print('CREATING FOLDER:', save_folder)
        os.makedirs(save_folder)

        print('SAVING:SCORES.PKL')
        with open(f"{save_folder}/results.pkl", "wb") as f:
            pickle.dump(results, f)
    return