import argparse
import os

import pandas as pd


RAW_COLUMNS = [
    "Frame",
    "Object",
    "Object ID",
    "Team",
    "X1",
    "Y1",
    "X2",
    "Y2",
    "X_Pitch",
    "Y_Pitch",
]


def read_tracking_csv(path):
    df = pd.read_csv(path)

    # Older notebook exports had a duplicate/wrong header:
    # Frame,Object,Object ID,Team,X1,Y1,X1,X2,X_Pitch,Y_Pitch
    # The values are still [X1, Y1, X2, Y2], so normalize by position.
    if len(df.columns) >= len(RAW_COLUMNS):
        df = df.iloc[:, : len(RAW_COLUMNS)].copy()
        df.columns = RAW_COLUMNS

    return df


def to_mot(df):
    players = df[df["Object"] == "player"].copy()

    numeric_columns = ["Frame", "Object ID", "Team", "X1", "Y1", "X2", "Y2"]
    for column in numeric_columns:
        players[column] = pd.to_numeric(players[column], errors="coerce")

    players = players.dropna(subset=["Frame", "Object ID", "X1", "Y1", "X2", "Y2"])
    players["w"] = players["X2"] - players["X1"]
    players["h"] = players["Y2"] - players["Y1"]
    players = players[(players["w"] > 0) & (players["h"] > 0)].copy()

    mot = pd.DataFrame(
        {
            0: players["Frame"].astype(int),
            1: players["Object ID"].astype(int),
            2: players["X1"].round(2),
            3: players["Y1"].round(2),
            4: players["w"].round(2),
            5: players["h"].round(2),
            6: 1.0,
            7: -1,
            8: -1,
            9: -1,
        }
    )

    return mot, players


def write_mot(path, mot):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mot.to_csv(path, index=False, header=False)
    print(f"Saved {len(mot)} rows: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FootballTrackingDataGeneration CSV to MOT txt."
    )
    parser.add_argument("input_csv", help="Tracking CSV exported from track.ipynb")
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence name for gta-link txt files. Defaults to input file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="mot_data/pred",
        help="Directory where MOT txt files are written.",
    )
    args = parser.parse_args()

    sequence = args.sequence or os.path.splitext(os.path.basename(args.input_csv))[0]

    df = read_tracking_csv(args.input_csv)
    mot_all, players = to_mot(df)

    write_mot(os.path.join(args.output_dir, "all", f"{sequence}.txt"), mot_all)

    for team_id, team_name in [(0, "home"), (1, "away")]:
        team_players = players[players["Team"].astype(int) == team_id].copy()
        team_mot, _ = to_mot(team_players)
        write_mot(
            os.path.join(args.output_dir, team_name, f"{sequence}.txt"),
            team_mot,
        )


if __name__ == "__main__":
    main()
