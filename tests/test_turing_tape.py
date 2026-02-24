"""
Comprehensive tests for TuringTape3D.
Covers: init, read/write, move (all directions + aliases), jump, scan,
serialization (to_dict/from_dict), persistence (save/load), repr, status,
edge cases, and error handling.
"""

import json
from pathlib import Path

import pytest

from turing_tape import TuringTape3D, DIRECTIONS, VALID_DIRECTIONS


class TestTuringTapeInit:
    def test_default_head_position(self, tape):
        assert tape.head == (0, 0, 0)

    def test_default_cells_empty(self, tape):
        assert tape.cells == {}

    def test_cells_returns_copy(self, tape):
        tape.write("x")
        cells = tape.cells
        cells[(99, 99, 99)] = "sneaky"
        assert (99, 99, 99) not in tape.cells


class TestTuringTapeRead:
    def test_read_empty_cell(self, tape):
        assert tape.read() == ""

    def test_read_written_cell(self, tape):
        tape.write("hello")
        assert tape.read() == "hello"

    def test_read_at_different_positions(self, tape):
        tape.write("origin")
        tape.jump(1, 0, 0)
        assert tape.read() == ""
        tape.jump(0, 0, 0)
        assert tape.read() == "origin"


class TestTuringTapeWrite:
    def test_write_returns_confirmation(self, tape):
        result = tape.write("test")
        assert "Wrote" in result
        assert "'test'" in result
        assert "(0, 0, 0)" in result

    def test_write_overwrites(self, tape):
        tape.write("first")
        tape.write("second")
        assert tape.read() == "second"

    def test_write_empty_string_clears_cell(self, tape):
        tape.write("data")
        assert tape.read() == "data"
        tape.write("")
        assert tape.read() == ""
        # Cell should be removed from dict entirely
        assert (0, 0, 0) not in tape._cells

    def test_write_non_string_raises(self, tape):
        with pytest.raises(TypeError, match="strings"):
            tape.write(42)

    def test_write_clears_cell_not_present(self, tape):
        # Writing empty to an already-empty cell should not error
        result = tape.write("")
        assert "Wrote" in result


class TestTuringTapeMove:
    @pytest.mark.parametrize("direction,expected", [
        ("+x", (1, 0, 0)),
        ("-x", (-1, 0, 0)),
        ("+y", (0, 1, 0)),
        ("-y", (0, -1, 0)),
        ("+z", (0, 0, 1)),
        ("-z", (0, 0, -1)),
    ])
    def test_move_all_directions(self, tape, direction, expected):
        result = tape.move(direction)
        assert tape.head == expected
        assert "Moved" in result

    @pytest.mark.parametrize("alias,expected", [
        ("up", (0, 1, 0)),
        ("down", (0, -1, 0)),
        ("left", (-1, 0, 0)),
        ("right", (1, 0, 0)),
        ("forward", (0, 0, 1)),
        ("backward", (0, 0, -1)),
        ("back", (0, 0, -1)),
    ])
    def test_move_aliases(self, tape, alias, expected):
        tape.move(alias)
        assert tape.head == expected

    def test_move_case_insensitive(self, tape):
        tape.move("UP")
        assert tape.head == (0, 1, 0)

    def test_move_strips_whitespace(self, tape):
        tape.move("  +x  ")
        assert tape.head == (1, 0, 0)

    def test_move_invalid_direction_raises(self, tape):
        with pytest.raises(ValueError, match="Invalid direction"):
            tape.move("diagonal")

    def test_sequential_moves(self, tape):
        tape.move("+x")
        tape.move("+y")
        tape.move("+z")
        assert tape.head == (1, 1, 1)

    def test_move_negative_positions(self, tape):
        tape.move("-x")
        tape.move("-y")
        tape.move("-z")
        assert tape.head == (-1, -1, -1)


class TestTuringTapeJump:
    def test_jump_to_position(self, tape):
        result = tape.jump(5, 10, -3)
        assert tape.head == (5, 10, -3)
        assert "Jumped" in result

    def test_jump_converts_to_int(self, tape):
        tape.jump(1.9, 2.1, 3.7)
        assert tape.head == (1, 2, 3)

    def test_jump_to_origin(self, tape):
        tape.jump(99, 99, 99)
        tape.jump(0, 0, 0)
        assert tape.head == (0, 0, 0)


class TestTuringTapeScan:
    def test_scan_empty(self, tape):
        result = tape.scan_neighborhood()
        assert result == {}

    def test_scan_finds_nearby(self, tape):
        tape.write("center")
        tape.jump(1, 0, 0)
        tape.write("right")
        tape.jump(0, 0, 0)  # Back to origin
        result = tape.scan_neighborhood(radius=1)
        assert "0,0,0" in result
        assert result["0,0,0"] == "center"
        assert "1,0,0" in result
        assert result["1,0,0"] == "right"

    def test_scan_excludes_far_cells(self, tape):
        tape.write("here")
        tape.jump(10, 10, 10)
        tape.write("far")
        tape.jump(0, 0, 0)
        result = tape.scan_neighborhood(radius=1)
        assert "10,10,10" not in result

    def test_scan_radius_zero(self, tape):
        tape.write("me")
        tape.jump(1, 0, 0)
        tape.write("not me")
        tape.jump(0, 0, 0)
        result = tape.scan_neighborhood(radius=0)
        assert "0,0,0" in result
        assert "1,0,0" not in result

    def test_scan_large_radius(self, tape):
        tape.write("a")
        tape.jump(3, 3, 3)
        tape.write("b")
        tape.jump(0, 0, 0)
        result = tape.scan_neighborhood(radius=5)
        assert "0,0,0" in result
        assert "3,3,3" in result

    def test_scan_negative_radius_raises(self, tape):
        with pytest.raises(ValueError, match="non-negative"):
            tape.scan_neighborhood(radius=-1)


class TestTuringTapeSerialization:
    def test_to_dict_empty(self, tape):
        d = tape.to_dict()
        assert d == {"head": [0, 0, 0], "cells": {}}

    def test_to_dict_with_data(self, tape):
        tape.write("val")
        tape.jump(1, 2, 3)
        d = tape.to_dict()
        assert d["head"] == [1, 2, 3]
        assert d["cells"]["0,0,0"] == "val"

    def test_from_dict_empty(self):
        tape = TuringTape3D.from_dict({"head": [0, 0, 0], "cells": {}})
        assert tape.head == (0, 0, 0)
        assert tape.cells == {}

    def test_from_dict_with_data(self):
        tape = TuringTape3D.from_dict({
            "head": [5, -3, 7],
            "cells": {"1,2,3": "hello", "0,0,0": "world"},
        })
        assert tape.head == (5, -3, 7)
        tape.jump(1, 2, 3)
        assert tape.read() == "hello"

    def test_roundtrip_serialization(self, tape):
        tape.write("origin")
        tape.jump(100, -50, 0)
        tape.write("far away")
        tape.move("+z")

        d = tape.to_dict()
        restored = TuringTape3D.from_dict(d)
        assert restored.head == tape.head
        assert restored.cells == tape.cells

    def test_from_dict_defaults(self):
        tape = TuringTape3D.from_dict({})
        assert tape.head == (0, 0, 0)
        assert tape.cells == {}


class TestTuringTapePersistence:
    def test_save_and_load(self, tape, tmp_path):
        tape.write("persisted")
        tape.jump(7, 8, 9)
        path = tmp_path / "tape.json"
        tape.save(path)

        loaded = TuringTape3D.load(path)
        assert loaded.head == (7, 8, 9)
        loaded.jump(0, 0, 0)
        assert loaded.read() == "persisted"

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        tape = TuringTape3D.load(path)
        assert tape.head == (0, 0, 0)
        assert tape.cells == {}

    def test_save_creates_valid_json(self, tape, tmp_path):
        tape.write("test")
        path = tmp_path / "tape.json"
        tape.save(path)
        data = json.loads(path.read_text())
        assert "head" in data
        assert "cells" in data


class TestTuringTapeDisplay:
    def test_repr(self, tape):
        r = repr(tape)
        assert "TuringTape3D" in r
        assert "head=(0, 0, 0)" in r
        assert "cells=0" in r

    def test_repr_with_data(self, tape):
        tape.write("x")
        r = repr(tape)
        assert "cells=1" in r

    def test_status(self, tape):
        s = tape.status()
        assert "Head: (0, 0, 0)" in s
        assert "Total written cells: 0" in s

    def test_status_with_data(self, tape):
        tape.write("val")
        s = tape.status()
        assert "'val'" in s
        assert "Total written cells: 1" in s


class TestDirectionsConstant:
    def test_all_six_directions_defined(self):
        assert len(DIRECTIONS) == 6

    def test_valid_directions_frozenset(self):
        assert isinstance(VALID_DIRECTIONS, frozenset)
        assert len(VALID_DIRECTIONS) == 6
