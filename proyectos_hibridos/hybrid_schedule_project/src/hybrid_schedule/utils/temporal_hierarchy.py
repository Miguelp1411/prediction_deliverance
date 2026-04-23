from __future__ import annotations


def macroblock_bins_from_minutes(bin_minutes: int, macroblock_minutes: int) -> int:
    if macroblock_minutes <= 0:
        raise ValueError('macroblock_minutes debe ser > 0')
    if macroblock_minutes % bin_minutes != 0:
        raise ValueError('macroblock_minutes debe ser múltiplo de bin_minutes')
    return macroblock_minutes // bin_minutes


def num_macroblocks_per_day(bins_per_day: int, macroblock_bins: int) -> int:
    if bins_per_day % macroblock_bins != 0:
        raise ValueError('bins_per_day debe ser divisible por macroblock_bins')
    return bins_per_day // macroblock_bins


def split_week_bin(start_bin: int, bins_per_day: int, macroblock_bins: int) -> tuple[int, int, int]:
    day_idx = int(start_bin // bins_per_day)
    in_day = int(start_bin % bins_per_day)
    macro_idx = int(in_day // macroblock_bins)
    fine_idx = int(in_day % macroblock_bins)
    return day_idx, macro_idx, fine_idx


def compose_week_bin(day_idx: int, macro_idx: int, fine_idx: int, bins_per_day: int, macroblock_bins: int) -> int:
    return int(day_idx) * bins_per_day + int(macro_idx) * macroblock_bins + int(fine_idx)
