"""The funding arithmetic in the 15 November project report.

Every figure here is an assumption. Nothing is a measurement and nothing is a forecast.
The report quotes these numbers, and the test checks that the quotes still add up.
"""
from __future__ import annotations

ASK = 2_000_000
PRICE_PER_GPU_YEAR = 1_500

# 18 months. The reserve is what is left after the named lines.
USE_OF_FUNDS = (
    ("Founder draw", 72_000),
    ("Runtime lead", 144_000),
    ("Systems engineer", 117_000),
    ("Applications engineer, from month 4", 90_000),
    ("Part-time finance and company secretary", 45_000),
    ("Equipment, working space, recruiting", 55_000),
    ("Graphics-board time, rental and one small server", 350_000),
    ("Patent counsel, through a PCT filing", 110_000),
    ("Formation, accounting, insurance, audit", 45_000),
    ("Two pilot deployments, travel and support", 60_000),
    ("Reserve, drawn quarterly against a written note", 912_000),
)

# End-of-year boards under license, years 1 to 15. An assumption, not a pipeline.
BASE_GPUS = (0, 25, 100, 250, 600, 1_200, 2_000, 3_000, 4_500, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000)


def use_total() -> int:
    return sum(amount for _, amount in USE_OF_FUNDS)


def revenue(gpus: int) -> float:
    """Revenue in millions of dollars."""
    return gpus * PRICE_PER_GPU_YEAR / 1_000_000


def operating_cost(revenue_m: float, year: int) -> float:
    """Millions. A floor that grows 12% a year, or 55% of revenue, whichever is larger."""
    floor = 0.80 * (1.12 ** (year - 1))
    return max(floor, 0.55 * revenue_m)


def exit_value(gpus_year15_index: int, multiple: float, ownership: float, year: int) -> float:
    """The investor's stake in millions, on a sale at `multiple` times that year's revenue."""
    gpus = BASE_GPUS[year - 1]
    if gpus_year15_index == 0:
        gpus //= 2
    elif gpus_year15_index == 2:
        gpus *= 3
    return revenue(gpus) * multiple * ownership


def main() -> None:
    assert use_total() == ASK
    print(f"use of funds ${use_total():,}")
    for year in (1, 3, 5, 10, 15):
        rev = revenue(BASE_GPUS[year - 1])
        cost = operating_cost(rev, year)
        print(f"year {year:2}  gpus {BASE_GPUS[year - 1]:6,}  revenue ${rev:6.2f}M  cost ${cost:6.2f}M")
    print("base year 10 stake", round(exit_value(1, 5, 0.12, 10), 2))
    print("base year 15 stake", round(exit_value(1, 5, 0.08, 15), 2))


if __name__ == "__main__":
    main()
