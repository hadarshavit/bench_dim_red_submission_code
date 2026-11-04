"""
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
        https://github.com/LeoIV/BAxUS/blob/main/baxus/benchmarks/real_world_benchmarks.py
"""


import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine
import uuid

import torch


DEVICE = 'cpu'
DTYPE = torch.float32

class Mopta08:

    def __init__(self):
        super().__init__()
        self.dim = 124
        self.lb=torch.zeros(self.dim, device=DEVICE, dtype=DTYPE)
        self.ub=torch.ones(self.dim, device=DEVICE, dtype=DTYPE)
        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(
            Path(__file__).parent, self._mopta_exectutable
        )

        self.directory_name = os.path.join(os.environ['TMP'], uuid.uuid4().hex)
        os.makedirs(self.directory_name, exist_ok=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Mopta08 benchmark for one point
        :param x: one input configuration
        :return: value with soft constraints
        """
        x = x.squeeze()
        assert x.ndim == 1

        Path(os.path.join(self.directory_name, 'input.txt')).parent.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x.detach().cpu().numpy()}\n")

        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()

        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = torch.tensor([float(x) for x in output if len(x) > 0], dtype=DTYPE, device=DEVICE)
        value = output[0]
        constraints = output[1:]

        return (value + 10 * torch.sum(torch.clip(constraints, min=0, max=None))).unsqueeze(-1)
