# SPDX-License-Identifier: Apache-2.0
"""Checked static bounds for compact XQA plans; no device reads."""

TILE = 256
MAX_BATCH = 32
MAX_CTAS = (1 << 31) - 1 - MAX_BATCH
PAGE_SIZES = (16, 32, 64)
HEAD_GROUP_RATIOS = (4, 6, 8)


def check_head_config(num_kv_heads: int = 1, head_group_ratio: int = 6) -> None:
    if type(num_kv_heads) is not int or not 1 <= num_kv_heads <= (1 << 31) - 1:
        raise ValueError("num_kv_heads must be a bounded positive integer")
    if type(head_group_ratio) is not int or head_group_ratio not in HEAD_GROUP_RATIOS:
        raise ValueError("head_group_ratio must be 4, 6 or 8")


def workspace_capacity(batch, ctas, table_width, page_size=32, num_kv_heads=1):
    check_head_config(num_kv_heads)
    capacity = plan_capacity(batch, ctas, table_width, page_size) * num_kv_heads
    if capacity > (1 << 31) - 1:
        raise ValueError("Total native scratch slots must fit int32")
    return capacity


def max_table_width(page_size: int = 32) -> int:
    if type(page_size) is not int or page_size not in PAGE_SIZES:
        raise ValueError("page_size must be 16, 32 or 64")
    return ((1 << 31) - 1 - (TILE - 1)) // page_size


MAX_TABLE_WIDTH = max_table_width()


def plan_capacity(batch: int, ctas: int, table_width: int, page_size: int = 32) -> int:
    """Capacity bound for intersections of B request intervals and C buckets."""
    if (
        type(batch) is not int
        or type(ctas) is not int
        or not 1 <= batch <= MAX_BATCH
        or not batch <= ctas <= MAX_CTAS
        or type(table_width) is not int
        or not 1 <= table_width <= max_table_width(page_size)
    ):
        raise ValueError(
            "Expected B in [1,32], bounded integer CTA capacity and table width"
        )
    return ctas + batch - 1


def plan_geometry(
    batch: int,
    table_width: int,
    sm_count: int,
    page_size: int = 32,
    num_kv_heads: int = 1,
) -> tuple[int, bool]:
    """Return checked CTA capacity and static direct-path eligibility."""
    plan_capacity(batch, batch, table_width, page_size)
    if type(sm_count) is not int or not 1 <= sm_count <= MAX_CTAS:
        raise ValueError("Expected a bounded positive integer SM count")
    check_head_config(num_kv_heads)
    if batch * num_kv_heads > (1 << 31) - 1:
        raise ValueError("Request/head counters must fit int32")
    # All inputs were checked before products or rounding. Live lengths stay
    # on device; a wide reserved table may therefore retain a larger grid.
    tiles_per_request = (table_width * page_size + TILE - 1) // TILE
    total_tiles = batch * tiles_per_request
    budget = max(1, (sm_count + num_kv_heads - 1) // num_kv_heads)
    return max(batch, min(budget, total_tiles)), batch == 1 or total_tiles <= budget
