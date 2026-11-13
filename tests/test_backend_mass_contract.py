import unittest
from pathlib import Path
from unittest.mock import patch

from lqcd_analysis.generation import Geometry
from lqcd_analysis.workflow import _check_xqcd_template


class BackendMassContractTests(unittest.TestCase):
    def test_extra_and_nonzero_mass_indices_rejected_before_any_backend(self):
        geometry = Geometry(4, 1, 1, 1)
        work = Path("virtual-new-workdir").resolve()
        backend = {"input_template": "virtual.template"}
        for text in ("masses:length 1\nmasses:0 0.400000\nmasses:1 0.2\n",
                     "masses:length 1\nmasses:1 0.400000\n",
                     "masses:length 2\nmasses:0 0.400000\nmasses:1 0.2\n"):
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                with self.assertRaisesRegex(ValueError, "exactly one mass"):
                    _check_xqcd_template(backend, geometry, "0.400000", "1", "gauge.fixture", work)


if __name__ == "__main__":
    unittest.main()
