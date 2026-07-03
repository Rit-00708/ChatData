"""
Dataset generator — creates demo CSV/Excel files for testing ChatData.

Generates an NBA 2024-25 player statistics dataset with:
- Realistic stats (points, rebounds, assists, steals, blocks, etc.)
- Some missing values and inconsistencies built in
- Categorical columns (team, position)
- Numeric outliers for cleaning detection demos
"""

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — teams, positions, names
# ---------------------------------------------------------------------------
TEAMS = [
    "ATL", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL",
    "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "UTA", "WAS", "BKN", "ORB",
]

POSITIONS = ["PG", "SG", "SF", "PF", "C"]

FIRST_NAMES = [
    "LeBron", "Stephen", "Kevin", "Giannis", "Nikola", "Luka",
    "Joel", "Jimmy", "Jayson", "Devin", "Shai", "Anthony",
    "Damian", "Donovan", "Bam", "Tyrese", "Jalen", "Paolo",
    "Kawhi", "Paul", "Draymond", "Kyrie", "Klay", "Pascal",
    "Domantas", "Jaren", "Victor", "LaMelo", "Tyler", "Cade",
    "Trae", "Dejounte", "Scottie", "Buddy", "Jalen", "Cason",
    "Jarrett", "Evan", "Darius", "Isaac", "Saddiq", "KZ",
    "Jericho", "Yves", "Maxi", "Immanuel", "Tari", "Aaron",
    "Omer", "Jabari", "Goga", "Mason", "Jordan", "Clint",
]

LAST_NAMES = [
    "James", "Curry", "Adams", "Antetokounmpo", "Jokic", "Doncic",
    "Embiid", "Butler", "Tatum", "Booker", "Holmgren", "Alexander",
    "Lowry", "Hernangomez", "Azubuike", "Banchi", "Diabate",
    "Kornher", "Jones", "Williams", "Rivers", "Mason", "Okogie",
    "Tucker", "Dort", "Giddey", "Caruso", "Wembanyama", "LaPonte",
    "Marjon", "Cunningham", "Young", "Reves", "Collins", "Gordon",
    "Bridges", "Washington", "Korban", "Ferguson", "McDaniels",
    "Ebiere", "Chiozza", "Clark", "Walker", "Henderson",
]


def _pick_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# ---------------------------------------------------------------------------
# Statistical models — positional baselines (per-game averages)
# ---------------------------------------------------------------------------
POSITION_BASALINES = {
    "PG": {"min_pts": 12, "max_pts": 28, "min_trb": 3, "max_trb": 7,
           "min_ast": 6, "max_ast": 14, "min_stl": 0.5, "max_stl": 2.0,
           "min_blk": 0.0, "max_blk": 0.5, "min_fg3m": 1.5, "max_fg3m": 4.0,
           "min_ftm": 1.0, "max_ftm": 4.0, "min_ftp": 72, "max_ftp": 90},
    "SG": {"min_pts": 14, "max_pts": 32, "min_trb": 2, "max_trb": 5,
           "min_ast": 2, "max_ast": 6, "min_stl": 0.5, "max_stl": 1.8,
           "min_blk": 0.0, "max_blk": 0.4, "min_fg3m": 2.5, "max_fg3m": 5.5,
           "min_ftm": 1.5, "max_ftm": 5.0, "min_ftp": 75, "max_ftp": 92},
    "SF": {"min_pts": 13, "max_pts": 28, "min_trb": 4, "max_trb": 7,
           "min_ast": 2, "max_ast": 5, "min_stl": 0.8, "max_stl": 1.5,
           "min_blk": 0.3, "max_blk": 0.8, "min_fg3m": 1.5, "max_fg3m": 4.0,
           "min_ftm": 2.0, "max_ftm": 6.0, "min_ftp": 70, "max_ftp": 88},
    "PF": {"min_pts": 12, "max_pts": 26, "min_trb": 6, "max_trb": 11,
           "min_ast": 1, "max_ast": 4, "min_stl": 0.5, "max_stl": 1.2,
           "min_blk": 0.3, "max_blk": 1.0, "min_fg3m": 0.8, "max_fg3m": 3.0,
           "min_ftm": 2.0, "max_ftm": 5.5, "min_ftp": 68, "max_ftp": 85},
    "C":  {"min_pts": 10, "max_pts": 24, "min_trb": 8, "max_trb": 14,
           "min_ast": 1, "max_ast": 3, "min_stl": 0.3, "max_stl": 0.8,
           "min_blk": 1.0, "max_blk": 3.0, "min_fg3m": 0.0, "max_fg3m": 1.5,
           "min_ftm": 2.0, "max_ftm": 6.0, "min_ftp": 60, "max_ftp": 82},
}


def _generate_player_stats(
    player_id: int,
    age: int | None,
) -> dict:
    """Generate a single row of NBA-style stats for one player."""
    pos = random.choice(POSITIONS)
    base = POSITION_BASALINES[pos]

    pts = round(random.uniform(base["min_pts"], base["max_pts"]), 1)
    trb = round(random.uniform(base["min_trb"], base["max_trb"]), 1)
    ast = round(random.uniform(base["min_ast"], base["max_ast"]), 1)
    stl = round(random.uniform(base["min_stl"], base["max_stl"]), 1)
    blk = round(random.uniform(base["min_blk"], base["max_blk"]), 1)

    # FG/3P/FT made + attempts, with consistency (made <= attempts)
    fga = max(pts / 0.45, pts + 2)  # ~45% FG estimate → get attempts
    fg_made = round(min(pts / (pts + 1) * random.uniform(0.38, 0.52), fga), 1)
    fg_pct = round(fg_made / max(fga, 1) * 100, 1)

    fg3a = random.randint(3, 12)
    fg3_made = round(min(random.uniform(base["min_fg3m"], base["max_fg3m"]), fg3a), 1)
    fg3_pct = round(fg3_made / max(fg3a, 1) * 100, 1)

    ft_a = random.randint(2, 10)
    ft_made = round(min(random.uniform(base["min_ftm"], base["max_ftm"]), ft_a), 1)
    ft_pct = round(ft_made / max(ft_a, 1) * 100, 1)

    tov = round(random.uniform(1.0, 4.5), 1)
    pf = round(random.uniform(1.0, 4.0), 1)
    games = random.randint(45, 82)
    gs = random.randint(max(0, games - 30), games)

    # Occasional outliers (5 % chance) — inflate pts or reb to crazy values
    if random.random() < 0.05:
        outlier_col = random.choice(["pts", "trb"])
        if outlier_col == "pts":
            pts = round(pts * random.uniform(1.6, 2.2), 1)
        else:
            trb = round(trb * random.uniform(1.8, 2.5), 1)

    return {
        "PlayerID": f"PLR-{player_id:04d}",
        "Name": _pick_name(),
        "Age": age if age is not None else "",
        "Team": random.choice(TEAMS),
        "Position": pos,
        "Games": games,
        "GamesStarted": gs,
        "MinutesPerGame": round(random.uniform(12.0, 38.0), 1),
        "PointsPerGame": pts,
        "ReboundsPerGame": trb,
        "AssistsPerGame": ast,
        "StealsPerGame": stl,
        "BlocksPerGame": blk,
        "FG3MadePerGame": fg3_made,
        "FG3AttemptsPerGame": round(fg3a, 1),
        "FG3Pct": fg3_pct,
        "FTPct": ft_pct,
        "TurnoversPerGame": tov,
        "PersonalFoulsPerGame": pf,
    }


def _validate_row(row: dict) -> bool:
    """Return True if row is valid (catches any impossible combos)."""
    if row["FG3MadePerGame"] > row["FG3AttemptsPerGame"]:
        return False
    if row["FG3Pct"] < 0 or row["FG3Pct"] > 100:
        return False
    if row["FTPct"] < 40 or row["FTPct"] > 100:
        return False
    if row["GamesStarted"] > row["Games"]:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_nba_dataset(
    n_players: int = 450,
    seed: int | None = 42,
) -> list[dict]:
    """Generate *n* rows of simulated NBA 2024-25 player statistics.

    Returns a list-of-dicts suitable for ``pd.DataFrame(rows).to_csv(...)``.
    About 3 % of rows have missing values; ~5 % have outlier stats — perfect
    for demonstrating the auto-cleaning pipeline.
    """
    if seed is not None:
        random.seed(seed)

    rows: list[dict] = []
    attempts = 0
    while len(rows) < n_players and attempts < n_players * 5:
        # Pick a random age (19-38), sometimes leave blank (~3 %)
        age = None
        if random.random() > 0.03:
            age = random.randint(19, 38)

        row = _generate_player_stats(len(rows) + 1, age)
        attempts += 1

        if not _validate_row(row):
            continue

        # ~2 % chance of blanking Name
        if random.random() < 0.02:
            row["Name"] = ""
        rows.append(row)

    return rows[:n_players]


def save_nba_dataset(
    output_dir: str | Path = "data",
    n_players: int = 450,
    seed: int = 42,
) -> list[Path]:
    """Generate the NBA dataset and save as CSV *and* Excel.

    Returns the list of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = generate_nba_dataset(n_players=n_players, seed=seed)

    import pandas as pd
    df = pd.DataFrame(rows)

    csv_path = out / "nba_2024_25_players.csv"
    df.to_csv(csv_path, index=False)

    xlsx_path = out / "nba_2024_25_players.xlsx"
    df.to_excel(xlsx_path, index=False)

    return [csv_path, xlsx_path]


if __name__ == "__main__":
    saved = save_nba_dataset()
    print(f"Generated NBA dataset → {saved}")
