# SPDX-License-Identifier: Apache-2.0
"""Checked static bounds for compact XQA plans; no device reads."""

TILE = 256
MAX_BATCH = 32
MAX_CTAS = (1 << 31) - 1 - MAX_BATCH
MAX_TABLE_WIDTH = ((1 << 31) - 1 - (TILE - 1)) // 32


def plan_capacity(batch: int, ctas: int, table_width: int) -> int:
    """Capacity bound for intersections of B request intervals and C buckets."""
    if (
        type(batch) is not int
        or type(ctas) is not int
        or not 1 <= batch <= MAX_BATCH
        or not batch <= ctas <= MAX_CTAS
        or type(table_width) is not int
        or not 1 <= table_width <= MAX_TABLE_WIDTH
    ):
        raise ValueError(
            "Expected B in [1,32], bounded integer CTA capacity and table width"
        )
    return ctas + batch - 1


def plan_ctas(batch: int, table_width: int, sm_count: int) -> int:
    """Cap cold capacity by table bounds while reserving zero-row owners."""
    plan_capacity(batch, batch, table_width)
    if type(sm_count) is not int or not 1 <= sm_count <= MAX_CTAS:
        raise ValueError("Expected a bounded positive integer SM count")
    # All inputs were checked before products or rounding. Live lengths stay
    # on device; a wide reserved table may therefore retain a larger grid.
    tiles_per_request = (table_width * 32 + TILE - 1) // TILE
    return max(batch, min(sm_count, batch * tiles_per_request))
