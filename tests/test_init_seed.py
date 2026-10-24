import ast
from pathlib import Path
import random as python_random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def load_init_seed(train_config):
    # Import only the seed dependencies and function: importing trainer.py
    # normally loads the dataset embeddings through config.configurator.
    source = Path(__file__).resolve().parents[1] / "encoder" / "trainer" / "trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected_imports = []
    init_seed_node = None
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name in {"random", "numpy"} for alias in node.names
        ):
            selected_imports.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy" and any(
            alias.name == "random" for alias in node.names
        ):
            selected_imports.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "init_seed":
            init_seed_node = node

    if init_seed_node is None:
        raise AssertionError("init_seed function not found")

    cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    cuda = SimpleNamespace(manual_seed=Mock(), manual_seed_all=Mock())
    torch_stub = SimpleNamespace(
        manual_seed=Mock(),
        cuda=cuda,
        backends=SimpleNamespace(cudnn=cudnn),
    )
    namespace = {
        "configs": {"train": train_config},
        "torch": torch_stub,
    }
    module = ast.Module(body=selected_imports + [init_seed_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["init_seed"], torch_stub


class InitSeedTests(unittest.TestCase):
    def test_reproducible_false_skips_seed_setup(self):
        python_random.seed(91)
        np.random.seed(91)
        python_state = python_random.getstate()
        numpy_state = np.random.get_state()
        init_seed, torch_stub = load_init_seed({"reproducible": False})

        init_seed()

        self.assertEqual(python_random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        torch_stub.manual_seed.assert_not_called()
        torch_stub.cuda.manual_seed.assert_not_called()
        torch_stub.cuda.manual_seed_all.assert_not_called()
        self.assertTrue(torch_stub.backends.cudnn.benchmark)
        self.assertFalse(torch_stub.backends.cudnn.deterministic)

    def test_reproducible_true_seeds_python_numpy_and_torch(self):
        seed = 2023
        expected_python = python_random.Random(seed).random()
        expected_numpy = np.random.RandomState(seed).random_sample()
        init_seed, torch_stub = load_init_seed({"reproducible": True, "seed": seed})

        init_seed()

        self.assertEqual(python_random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch_stub.manual_seed.assert_called_once_with(seed)
        torch_stub.cuda.manual_seed.assert_called_once_with(seed)
        torch_stub.cuda.manual_seed_all.assert_called_once_with(seed)
        self.assertFalse(torch_stub.backends.cudnn.benchmark)
        self.assertTrue(torch_stub.backends.cudnn.deterministic)


if __name__ == "__main__":
    unittest.main()
