import os
import tempfile
import unittest

from config import CFG
from io_files import write_coords, write_layout_view_html
from models import Placed, Rect


class WriteOutputsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._orig_coords = CFG.COORDS_OUT
        self._orig_layout = CFG.LAYOUT_HTML

    def tearDown(self) -> None:
        CFG.COORDS_OUT = self._orig_coords
        CFG.LAYOUT_HTML = self._orig_layout

    def test_write_coords_uses_configured_relative_path(self) -> None:
        CFG.COORDS_OUT = "outputs/custom_coords.txt"
        rect = Rect(2, 2, "tile")
        placed = [Placed(0, 0, rect)]

        path = write_coords(placed, 4, 4, self.tmpdir.name)

        expected = os.path.join(self.tmpdir.name, "outputs", "custom_coords.txt")
        self.assertEqual(path, expected)
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as fh:
            contents = fh.read()
        self.assertIn("tile", contents)

    def test_write_layout_view_html_accepts_absolute_path(self) -> None:
        target = os.path.join(self.tmpdir.name, "html", "layout.html")
        CFG.LAYOUT_HTML = target

        svg = "<svg></svg>"
        legend = "<li>tile</li>"

        path = write_layout_view_html(svg, legend, self.tmpdir.name)

        self.assertEqual(path, target)
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as fh:
            contents = fh.read()
        self.assertIn(svg, contents)
        self.assertIn(legend, contents)


if __name__ == "__main__":
    unittest.main()
