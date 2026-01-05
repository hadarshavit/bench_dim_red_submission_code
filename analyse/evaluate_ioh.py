from ioh import get_problem, ProblemType, ProblemClass
from concurrent.futures import ProcessPoolExecutor
import pickle


def get_problem_opt(fid, iid, dim):
    print(f"Getting {fid} {iid} {dim}", flush=True)
    return get_problem(fid, iid, dim, ProblemClass.BBOB).optimum.y


if __name__ == '__main__':
    fs = dict()

    with ProcessPoolExecutor(max_workers=96) as executor:
        for fid in range(1, 25):
            for iid in range(31):
                for dim in [20, 40, 60, 120, 240, 500, 1000]:
                    fs[(fid, iid, dim)] = executor.submit(get_problem_opt, fid, iid, dim)

    for fid in range(1, 25):
        for iid in range(31):
            for dim in [20, 40, 60, 120, 240, 500, 1000]:
                fs[(fid, iid, dim)] = fs[(fid, iid, dim)].result()

    with open('/benchmark_dim_red/analyse/optima.pkl', 'wb') as file:
        pickle.dump(fs, file)
    
