import argparse
import utils as utils


def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # split proportion we are using
    parser.add_argument("-split_select", required=True, nargs='?')
    # what selection scheme are we using
    parser.add_argument("-scheme", required=True, nargs='?')
    # what openml task are we using
    parser.add_argument("-task_id", required=True, nargs='?')
    # number of threads to use during estimator evalutation
    parser.add_argument("-n_jobs",  required=True, nargs='?')
    # where to save the results/models
    parser.add_argument("-savepath", required=True, nargs='?')
    # seed offset
    parser.add_argument("-seed", required=True, nargs='?')
    # is this a classification (True) or regression (False) task
    parser.add_argument("-task_type", required=True, nargs='?')

    args = parser.parse_args()
    split_select = float(args.split_select)
    print('Split:', split_select)
    scheme = str(args.scheme)
    print('Scheme:', scheme)
    task_id = int(args.task_id)
    print('Task ID:', task_id)
    n_jobs = int(args.n_jobs)
    print('Number of Jobs:', n_jobs)
    save_path = str(args.savepath)
    print('Save Path:', save_path)
    seed = int(args.seed)
    print('Seed:', seed)
    task_type = bool(int(args.task_type))
    # if task_type is True, then we are doing classification
    if task_type:
        print('Task Type: Classification')
    else:
        print('Task Type: Regression')

    # regression tasks tbd
    regression_tasks = []

    # Classification tasks from the 'AutoML Benchmark All Classification' suite
    # Suite is used within 'AMLB: an AutoML Benchmark' paper
    # https://github.com/openml/automlbenchmark
    # https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf
    # https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271

    # classification tasks:
    # 100 < rows < 2000
    # columns < 200
    classification_tasks = [146818,359954,359955,190146,168757,359956,
                            359958,359959,2073,359960,168784,359962]

    assert task_id in regression_tasks + classification_tasks, 'Task ID not in list of tasks'

    # execute task
    utils.execute_experiment(split_select,
                                      scheme,
                                      task_id,
                                      n_jobs,
                                      save_path,
                                      seed,
                                      task_type)

if __name__ == '__main__':
    main()
    print('FINISHED')