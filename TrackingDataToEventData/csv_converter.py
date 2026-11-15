import pandas as pd
import os
import math
import argparse

# ================= CONFIG =================
INPUT_CSV = "input_csv/2e57b9_0_1.csv"

OUTDIR = "data/Sample_Game_5"

FPS = 25
# Must match sports.configs.soccer.SoccerPitchConfiguration used in tracking.
PITCH_LENGTH = 12000
PITCH_WIDTH = 7000

MAX_PLAYERS_PER_TEAM = 14

HOME_TEAM_ID = 0
AWAY_TEAM_ID = 1
# ==========================================


def clean_raw_csv(path):
    raw = pd.read_csv(path)

    # Fix bad duplicate/wrong header from your original CSV
    raw.columns = [
        "Frame", "Object", "Object ID", "Team",
        "X1", "Y1", "X2", "Y2",
        "X_Pitch", "Y_Pitch"
    ]

    raw["Frame"] = raw["Frame"].astype(int)

    return raw


def normalize_coordinates(df):
    df["x"] = df["X_Pitch"] / PITCH_LENGTH
    df["y"] = df["Y_Pitch"] / PITCH_WIDTH
    return df


def force_majority_team(players):
    """
    ByteTrack IDs can sometimes get assigned to different teams
    across frames because the team classifier changes its prediction.

    This function forces each Object ID to belong to only one team:
    the team it was assigned to most often.
    """

    players["Object ID"] = players["Object ID"].astype(int)
    players["Team"] = players["Team"].astype(int)

    majority_team = (
        players
        .groupby("Object ID")["Team"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    players["Team"] = players["Object ID"].map(majority_team)

    return players


def get_main_player_ids(team_players, max_players=14):
    """
    Keep only the most frequent tracking IDs.

    This prevents temporary/false/reappearing ByteTrack IDs from being
    treated as real players.
    """

    counts = team_players["Object ID"].astype(int).value_counts()
    main_ids = counts.head(max_players).index.tolist()

    return sorted(main_ids)


def make_metrica_team(players, ball, frames, team_id, team_name, max_players=MAX_PLAYERS_PER_TEAM):
    team_players = players[players["Team"] == team_id].copy()
    team_players["Object ID"] = team_players["Object ID"].astype(int)

    player_ids = get_main_player_ids(team_players, max_players)

    print(f"{team_name} kept players:", player_ids)
    print(f"{team_name} player frame counts:")
    print(team_players["Object ID"].value_counts().head(20))
    print()

    row1 = ["", "", ""]
    row2 = ["", "", ""]
    row3 = ["Period", "Frame", "Time [s]"]

    for pid in player_ids:
        row1 += [team_name, ""]
        row2 += [str(pid), ""]
        row3 += [f"Player{pid}", ""]

    row1 += ["", ""]
    row2 += ["", ""]
    row3 += ["Ball", ""]

    rows = [row1, row2, row3]

    for frame_num in frames:
        row = [1, frame_num, frame_num / FPS]

        for pid in player_ids:
            p = team_players[
                (team_players["Frame"] == frame_num) &
                (team_players["Object ID"] == pid)
            ]

            if len(p) > 0:
                row += [p.iloc[0]["x"], p.iloc[0]["y"]]
            else:
                row += [math.nan, math.nan]

        b = ball[ball["Frame"] == frame_num]

        if len(b) > 0:
            row += [b.iloc[0]["x"], b.iloc[0]["y"]]
        else:
            row += [math.nan, math.nan]

        rows.append(row)

    return rows


def infer_game_id(outdir):
    dirname = os.path.basename(os.path.normpath(outdir))
    prefix = "Sample_Game_"
    if dirname.startswith(prefix):
        return int(dirname[len(prefix):])
    raise ValueError(
        "Could not infer game id from outdir. Pass --game-id explicitly."
    )


def main(input_csv=INPUT_CSV, outdir=OUTDIR, game_id=None, max_players=MAX_PLAYERS_PER_TEAM):
    os.makedirs(outdir, exist_ok=True)

    raw = clean_raw_csv(input_csv)

    players = raw[raw["Object"] == "player"].copy()
    ball = raw[raw["Object"] == "ball"].copy()

    players = force_majority_team(players)

    players = normalize_coordinates(players)
    ball = normalize_coordinates(ball)

    # Treat missing ball detection as NaN instead of real 0,0 location
    ball.loc[
        (ball["X_Pitch"] == 0) & (ball["Y_Pitch"] == 0),
        ["x", "y"]
    ] = math.nan

    frames = sorted(raw["Frame"].unique())

    home_rows = make_metrica_team(
        players=players,
        ball=ball,
        frames=frames,
        team_id=HOME_TEAM_ID,
        team_name="Home",
        max_players=max_players
    )

    away_rows = make_metrica_team(
        players=players,
        ball=ball,
        frames=frames,
        team_id=AWAY_TEAM_ID,
        team_name="Away",
        max_players=max_players
    )

    if game_id is None:
        game_id = infer_game_id(outdir)

    home_path = f"{outdir}/Sample_Game_{game_id}_RawTrackingData_Home_Team.csv"
    away_path = f"{outdir}/Sample_Game_{game_id}_RawTrackingData_Away_Team.csv"

    pd.DataFrame(home_rows).to_csv(home_path, index=False, header=False)
    pd.DataFrame(away_rows).to_csv(away_path, index=False, header=False)

    home_ids = set(players[players["Team"] == HOME_TEAM_ID]["Object ID"].unique())
    away_ids = set(players[players["Team"] == AWAY_TEAM_ID]["Object ID"].unique())
    overlap = home_ids & away_ids

    print("Saved:")
    print(home_path)
    print(away_path)
    print()
    print("Home/Away ID overlap:", overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert long tracking CSV to Metrica-style Home/Away files."
    )
    parser.add_argument(
        "--input-csv",
        default=INPUT_CSV,
        help="Long tracking CSV from the tracking notebook.",
    )
    parser.add_argument(
        "--outdir",
        default=OUTDIR,
        help="Output Sample_Game directory.",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Sample game id used in output filenames.",
    )
    parser.add_argument(
        "--max-players-per-team",
        type=int,
        default=MAX_PLAYERS_PER_TEAM,
        help="Number of most frequent player IDs to keep per team.",
    )
    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        outdir=args.outdir,
        game_id=args.game_id,
        max_players=args.max_players_per_team,
    )
