import numpy as np
import LassoBench
from .mujoco_bench import func_factories
from .svm import SVM as SVMBench
from .mopta import Mopta08
from .rover import RoverBenchmark


class LassoSynthetic:
    def __init__(self, bench_name) -> None:
        self.synt_bench = LassoBench.SyntheticBenchmark(pick_bench=bench_name)

    def __call__(self, x: np.array):
        return np.array([self.synt_bench.evaluate(x.numpy().astype(np.float64))])

    def get_bounds(self):
        return np.full(self.synt_bench.n_features, -1), np.full(self.synt_bench.n_features, 1)


class LassoReal:
    def __init__(self, bench_name) -> None:
        self.real_bench = LassoBench.RealBenchmark(pick_data=bench_name)

    def __call__(self, x: np.array):
        return np.array([self.real_bench.evaluate(x.numpy().astype(np.float64))])

    def get_bounds(self):
        return np.full(self.real_bench.n_features, -1), np.full(self.real_bench.n_features, 1)

class Mujoco:
    def __init__(self, bench_name) -> None:
        self.mujoco_bench = func_factories[bench_name].make_object()

    def __call__(self, x: np.array):
        return self.mujoco_bench(x.unsqueeze(0))[0]

    def get_bounds(self):
        return self.mujoco_bench.lb, self.mujoco_bench.ub


class SVM:
    def __init__(self) -> None:
        self.svm_bench = SVMBench()

    def __call__(self, x: np.array):
        return self.svm_bench(x).numpy()

    def get_bounds(self):
        return self.svm_bench.lb.numpy(), self.svm_bench.ub.numpy()


class Mopta:
    def __init__(self) -> None:
        self.mopta_bench = Mopta08()

    def __call__(self, x: np.array):
        return self.mopta_bench(x).numpy()

    def get_bounds(self):
        return self.mopta_bench.lb.numpy(), self.mopta_bench.ub.numpy()


class Rover:
    def __init__(self) -> None:
        self.rover_bench = RoverBenchmark()

    def __call__(self, x: np.array):
        return self.rover_bench(x).numpy()

    def get_bounds(self):
        return self.rover_bench.lb.numpy(), self.rover_bench.ub.numpy()


if __name__ == '__main__':
    from functools import partial
    benchs = [
        Rover, SVM,
             partial(Mujoco, 'ant'),
               partial(Mujoco, 'half_cheetah'),
              partial(Mujoco, 'hopper'), partial(Mujoco, 'humanoid'), partial(Mujoco, 'swimmer'),
              partial(Mujoco, 'walker_2d'),
            Mopta,
              partial(LassoReal, 'diabetes'), partial(LassoReal, 'breast_cancer'),
              partial(LassoReal, 'leukemia'), partial(LassoReal, 'dna'),
              partial(LassoReal, 'rcv1'), partial(LassoSynthetic, 'synt_simple'),
              partial(LassoSynthetic, 'synt_medium'), partial(LassoSynthetic, 'synt_high'),
              partial(LassoSynthetic, 'synt_hard')
              ]
    for b in benchs:
        print(b)
        bench = b()
        lb, ub = bench.get_bounds()
        print(type(lb))
        print(ub.shape)
        from sampling import create_initial_sample
        import torch
        random_config = torch.from_numpy(create_initial_sample(len(lb), sample_type = 'lhs', n=1, lower_bound=lb, upper_bound=ub, seed=42).iloc[0].values)


        print(random_config.shape)
        print(bench(random_config))
