"""Numeric grid ownership and candidate buffering; never operate on files."""
import math

import numpy as np
import pytest

from kon.desktop import DesktopTarget
from kon.grid_selection import GridSelector

SIZE = (3000, 1200)  # Default cells are exactly 100x100px for boundary tests.


def target(identity, center, *, radius=40):
    return DesktopTarget(identity, identity + ".txt", center, radius=radius)


def test_default_grid_matches_thirty_columns_twelve_rows_and_screen_edges():
    grid = GridSelector()
    assert (grid.columns, grid.rows, grid.hysteresis_pixels) == (30, 12, 8)
    assert grid.cell_for((0, 0), SIZE) == (0, 0)
    assert grid.cell_for((100, 100), SIZE) == (1, 1)
    assert grid.cell_for((2999, 1199), SIZE) == (29, 11)
    assert grid.cell_for((3000, 1200), SIZE) == (29, 11)
    assert grid.cell_for((-1, 0), SIZE) is None
    assert grid.cell_for((3000.001, 0), SIZE) is None
    assert grid.cell_bounds((29, 11), SIZE) == pytest.approx((2900, 1100, 100, 100))


@pytest.mark.parametrize("size", [(3440, 1440), (1920, 1080), (1000, 700)])
def test_all_cell_corners_have_consistent_bounds_and_membership_on_real_display_sizes(size):
    grid = GridSelector()
    for column in range(30):
        for row in range(12):
            x, y, width, height = grid.cell_bounds((column, row), size)
            assert grid.cell_for((x, y), size) == (column, row)
            assert grid.cell_for((x + width / 2, y + height / 2), size) == (column, row)
    assert grid.cell_bounds((0, 0), size)[2:] == pytest.approx((size[0] / 30, size[1] / 12))


def test_unique_file_owns_its_whole_cell_independently_of_old_icon_radius():
    grid = GridSelector()
    file = target("a", (5, 5), radius=1)
    assert grid.pick([file], (99, 99), SIZE) is file
    assert grid.active_cell == (0, 0)
    assert grid.ambiguous_cell is None


@pytest.mark.parametrize("delta", [-1e-16, 0., 1e-16])
def test_equivalent_normalized_boundary_pixels_snap_roundoff_on_first_pick(delta):
    grid = GridSelector(hysteresis_pixels=0)
    size = (3440, 1440)
    point = ((.5 + delta) * size[0], (.5 + delta) * size[1])
    file = target("center", (size[0] * .5, size[1] * .5))
    assert grid.cell_for(point, size) == (15, 6)
    assert grid.pick([file], point, size) is file


def test_actual_one_millionth_cell_difference_is_not_hidden_by_roundoff_snap():
    grid = GridSelector(hysteresis_pixels=0)
    size = (3440, 1440)
    before = ((.5 - 1e-6 / 30) * size[0], (.5 - 1e-6 / 12) * size[1])
    after = ((.5 + 1e-6 / 30) * size[0], (.5 + 1e-6 / 12) * size[1])
    assert grid.cell_for(before, size) == (14, 5)
    assert grid.cell_for(after, size) == (15, 6)
    file = target("center", (size[0] * .5, size[1] * .5))
    assert grid.pick([file], before, size) is None


def test_crossing_between_unique_file_cells_requires_exiting_eight_pixel_buffer():
    grid = GridSelector()
    first, second = target("a", (50, 50)), target("b", (150, 50))
    assert grid.pick([first, second], (95, 50), SIZE) is first
    for x in (100, 101, 107.99, 108):
        assert grid.cell_for((x, 50), SIZE) == (1, 0)
        assert grid.pick([first, second], (x, 50), SIZE) is first
        assert grid.active_cell == (0, 0)
    assert grid.pick([first, second], (108.001, 50), SIZE) is second
    assert grid.active_cell == (1, 0)
    assert grid.pick([first, second], (99, 50), SIZE) is second
    assert grid.pick([first, second], (92, 50), SIZE) is second
    assert grid.pick([first, second], (91.999, 50), SIZE) is first


def test_empty_neighbor_retains_candidate_only_inside_authorized_boundary_buffer():
    grid = GridSelector()
    file = target("a", (50, 50))
    assert grid.pick([file], (95, 50), SIZE) is file
    assert grid.pick([file], (108, 50), SIZE) is file
    assert grid.active_cell == (0, 0) and grid.ambiguous_cell is None
    assert grid.pick([file], (108.001, 50), SIZE) is None
    assert grid.active_cell is None
    # Moving back into an empty neighbor does not reacquire a distant file.
    assert grid.pick([file], (101, 50), SIZE) is None
    assert grid.pick([file], (99, 50), SIZE) is file


def test_diagonal_neighbor_uses_the_expanded_rectangle_on_both_axes():
    grid = GridSelector()
    first, diagonal = target("a", (50, 50)), target("d", (150, 150))
    assert grid.pick([first, diagonal], (90, 90), SIZE) is first
    assert grid.pick([first, diagonal], (108, 108), SIZE) is first
    assert grid.pick([first, diagonal], (108.001, 108), SIZE) is diagonal
    assert grid.active_cell == (1, 1)


def test_zero_hysteresis_switches_at_the_actual_cell_boundary():
    grid = GridSelector(hysteresis_pixels=0)
    first, second = target("a", (50, 50)), target("b", (150, 50))
    assert grid.pick([first, second], (99, 50), SIZE) is first
    assert grid.pick([first, second], (100, 50), SIZE) is second


def test_multiple_files_in_actual_cursor_cell_never_pick_arbitrarily():
    grid = GridSelector()
    first, second = target("a", (10, 10)), target("b", (90, 90))
    for point in ((10, 10), (90, 90), (50, 50)):
        assert grid.pick([first, second], point, SIZE) is None
        assert grid.ambiguous_cell == (0, 0)
        assert grid.active_cell is None


def test_collision_cell_under_cursor_overrides_hysteresis_immediately():
    grid = GridSelector()
    first, second, third = target("a", (50, 50)), target("b", (150, 50)), target("c", (160, 55))
    assert grid.pick([first, second], (99, 50), SIZE) is first
    assert grid.pick([first, second, third], (101, 50), SIZE) is None
    assert grid.active_cell is None and grid.ambiguous_cell == (1, 0)
    # Removing the ambiguity starts the actual cell, not a stale old buffer.
    assert grid.pick([first, second], (101, 50), SIZE) is second
    assert grid.ambiguous_cell is None and grid.active_cell == (1, 0)


def test_removed_or_newly_colliding_old_candidate_cannot_survive_boundary_buffer():
    first, second, third = target("a", (50, 50)), target("b", (150, 50)), target("c", (60, 50))
    grid = GridSelector()
    grid.pick([first, second], (99, 50), SIZE)
    assert grid.pick([second], (101, 50), SIZE) is second
    grid.reset()
    grid.pick([first, second], (99, 50), SIZE)
    assert grid.pick([first, second, third], (101, 50), SIZE) is second
    assert grid.active_cell == (1, 0) and grid.ambiguous_cell is None


def test_moved_file_loses_old_cell_ownership_and_returns_current_record_only():
    grid = GridSelector()
    original = target("a", (50, 50))
    moved = target("a", (150, 50))
    assert grid.pick([original], (99, 50), SIZE) is original
    assert grid.pick([moved], (99, 50), SIZE) is None
    assert grid.active_cell is None
    assert grid.pick([moved], (150, 50), SIZE) is moved
    updated_same_cell = target("a", (155, 55))
    assert grid.pick([updated_same_cell], (101, 50), SIZE) is updated_same_cell


def test_display_resize_discards_prior_hysteresis_state():
    grid = GridSelector()
    first, second = target("a", (50, 50)), target("b", (150, 50))
    assert grid.pick([first, second], (95, 50), SIZE) is first
    # 90px cells put x=95 in B's cell, still within A's hypothetical 8px
    # buffer. Resize must start fresh and choose the actual pointed cell.
    assert grid.pick([first, second], (95, 50), (2700, 1200)) is second
    assert grid.active_cell == (1, 0)


def test_duplicates_of_same_file_do_not_create_fake_collision_but_contradictory_records_fail():
    grid = GridSelector()
    first = target("a", (50, 50))
    duplicate = target("a", (50, 50))
    assert grid.pick([first, duplicate], (50, 50), SIZE) is first
    assert grid.ambiguous_cell is None
    contradictory = target("a", (150, 50))
    assert grid.pick([first, contradictory], (50, 50), SIZE) is None
    assert grid.active_cell is None and grid.ambiguous_cell is None


def test_offscreen_files_do_not_own_a_screen_cell():
    grid = GridSelector()
    outside = target("outside", (-1, 50))
    inside = target("a", (50, 50))
    assert grid.pick([outside, inside], (50, 50), SIZE) is inside
    assert grid.pick([outside], (50, 50), SIZE) is None


@pytest.mark.parametrize("point", [None, [], (1,), (1, 2, 3), (True, 50), ("50", 50),
                                  (math.nan, 50), (math.inf, 50), {"x": 50, "y": 50},
                                  (-.1, 50), (3000.1, 50), (50, 1200.1)])
def test_invalid_missing_or_offscreen_point_clears_old_candidate(point):
    grid = GridSelector()
    file = target("a", (50, 50))
    grid.pick([file], (50, 50), SIZE)
    assert grid.pick([file], point, SIZE) is None
    assert grid.active_cell is None and grid.ambiguous_cell is None


@pytest.mark.parametrize("size", [None, [], (3000,), (0, 1200), (-1, 1200), (3000, math.nan),
                                 (math.inf, 1200), (True, 1200), ("3000", 1200)])
def test_invalid_display_size_has_no_geometry_or_candidate(size):
    grid = GridSelector()
    file = target("a", (50, 50))
    grid.pick([file], (50, 50), SIZE)
    assert grid.cell_for((50, 50), size) is None
    assert grid.cell_bounds((0, 0), size) is None
    assert grid.pick([file], (50, 50), size) is None
    assert grid.active_cell is None


@pytest.mark.parametrize("cell", [None, [], (0,), (0, 0, 0), (-1, 0), (30, 0), (0, 12),
                                 (True, 0), (0., 0), ("0", 0), {"column": 0, "row": 0}])
def test_invalid_cell_bounds_fail_without_mutating_candidate(cell):
    grid = GridSelector()
    file = target("a", (50, 50))
    grid.pick([file], (50, 50), SIZE)
    assert grid.cell_bounds(cell, SIZE) is None
    assert grid.active_cell == (0, 0)


@pytest.mark.parametrize("targets", [None, [None], [object()], [target("a", (math.nan, 50))],
                                    [target("a", ("50", 50))], [target("", (50, 50))]])
def test_malformed_target_metadata_clears_old_candidate(targets):
    grid = GridSelector()
    file = target("a", (50, 50))
    grid.pick([file], (50, 50), SIZE)
    assert grid.pick(targets, (50, 50), SIZE) is None
    assert grid.active_cell is None and grid.ambiguous_cell is None


def test_empty_list_and_explicit_reset_clear_active_and_ambiguous_state():
    grid = GridSelector()
    first, second = target("a", (50, 50)), target("b", (60, 50))
    grid.pick([first], (50, 50), SIZE)
    assert grid.pick([], (50, 50), SIZE) is None
    assert grid.active_cell is None
    grid.pick([first, second], (50, 50), SIZE)
    assert grid.ambiguous_cell == (0, 0)
    grid.reset()
    assert grid.active_cell is None and grid.ambiguous_cell is None


def test_numpy_numeric_coordinates_and_indices_are_supported():
    grid = GridSelector(columns=np.int64(30), rows=np.int64(12))
    assert grid.cell_for(np.array([150., 250.]), np.array(SIZE)) == (1, 2)
    assert grid.cell_bounds((np.int64(1), np.int64(2)), SIZE) == (100, 200, 100, 100)


@pytest.mark.parametrize("kwargs", [{"columns": 0}, {"columns": 30.0}, {"columns": True},
                                   {"rows": -1}, {"rows": "12"}, {"rows": False},
                                   {"hysteresis_pixels": -1}, {"hysteresis_pixels": math.nan},
                                   {"hysteresis_pixels": math.inf}, {"hysteresis_pixels": "8"},
                                   {"hysteresis_pixels": True}])
def test_invalid_grid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GridSelector(**kwargs)
