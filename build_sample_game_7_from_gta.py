import math
import os
from collections import defaultdict, deque

import pandas as pd


RAW_TRACKING_CSV = "2e57b9_0_1.csv"
GAME_ID = 7
OUTDIR = f"TrackingDataToEventData/data/Sample_Game_{GAME_ID}"

HOME_REFINED_MOT = (
    "mot_data/pred/"
    "home_data_Split+Connect_eps0.6_minSamples10_K3_mergeDist0.4_spatial1.0/"
    "2e57b9_0_1.txt"
)
AWAY_REFINED_MOT = (
    "mot_data/pred/"
    "away_data_Split+Connect_eps0.6_minSamples10_K3_mergeDist0.4_spatial1.0/"
    "2e57b9_0_1.txt"
)

FPS = 25
PITCH_LENGTH = 12000
PITCH_WIDTH = 7000

HOME_TEAM_ID = 0
AWAY_TEAM_ID = 1

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

MOT_COLUMNS = [
    "Frame",
    "Refined ID",
    "X1",
    "Y1",
    "Width",
    "Height",
    "Confidence",
    "Unused1",
    "Unused2",
    "Unused3",
]


def read_tracking_csv(path):
    raw = pd.read_csv(path)
    raw = raw.iloc[:, : len(RAW_COLUMNS)].copy()
    raw.columns = RAW_COLUMNS

    for column in ["Frame", "Object ID", "Team", "X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["Frame"] = raw["Frame"].astype(int)
    return raw


def read_refined_mot(path):
    mot = pd.read_csv(path, names=MOT_COLUMNS)

    for column in ["Frame", "Refined ID", "X1", "Y1", "Width", "Height"]:
        mot[column] = pd.to_numeric(mot[column], errors="coerce")

    mot = mot.dropna(subset=["Frame", "Refined ID", "X1", "Y1", "Width", "Height"]).copy()
    mot["Frame"] = mot["Frame"].astype(int)
    mot["Refined ID"] = mot["Refined ID"].astype(int)
    mot["X2"] = mot["X1"] + mot["Width"]
    mot["Y2"] = mot["Y1"] + mot["Height"]
    return mot


def bbox_key(row):
    return (
        int(row["Frame"]),
        round(float(row["X1"]), 2),
        round(float(row["Y1"]), 2),
        round(float(row["X2"]), 2),
        round(float(row["Y2"]), 2),
    )


def map_refined_ids(players, refined_mot, team_id):
    team_players = players[players["Team"].astype(int) == team_id].copy()
    original_by_bbox = defaultdict(deque)

    for idx, row in team_players.iterrows():
        original_by_bbox[bbox_key(row)].append(idx)

    refined_ids = {}
    unmatched = []

    for _, row in refined_mot.iterrows():
        key = bbox_key(row)
        if original_by_bbox[key]:
            original_idx = original_by_bbox[key].popleft()
            refined_ids[original_idx] = int(row["Refined ID"])
        else:
            unmatched.append(key)

    return refined_ids, unmatched


def normalize_coordinates(df):
    df = df.copy()
    df["x"] = df["X_Pitch"] / PITCH_LENGTH
    df["y"] = df["Y_Pitch"] / PITCH_WIDTH
    return df


def make_metrica_team(players, ball, frames, team_id, team_name):
    team_players = players[players["Team"].astype(int) == team_id].copy()
    player_ids = [int(player_id) for player_id in sorted(team_players["Refined ID"].dropna().astype(int).unique())]

    row1 = ["", "", ""]
    row2 = ["", "", ""]
    row3 = ["Period", "Frame", "Time [s]"]

    for player_id in player_ids:
        row1 += [team_name, ""]
        row2 += [str(player_id), ""]
        row3 += [f"Player{player_id}", ""]

    row1 += ["", ""]
    row2 += ["", ""]
    row3 += ["Ball", ""]

    rows = [row1, row2, row3]

    grouped_players = {
        (int(row["Frame"]), int(row["Refined ID"])): row
        for _, row in team_players.dropna(subset=["Refined ID"]).iterrows()
    }
    grouped_ball = {int(row["Frame"]): row for _, row in ball.iterrows()}

    for frame_num in frames:
        row = [1, frame_num, frame_num / FPS]

        for player_id in player_ids:
            player = grouped_players.get((frame_num, player_id))
            if player is None:
                row += [math.nan, math.nan]
            else:
                row += [player["x"], player["y"]]

        frame_ball = grouped_ball.get(frame_num)
        if frame_ball is None:
            row += [math.nan, math.nan]
        else:
            row += [frame_ball["x"], frame_ball["y"]]

        rows.append(row)

    return rows, player_ids


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    raw = read_tracking_csv(RAW_TRACKING_CSV)
    players = raw[raw["Object"] == "player"].copy()
    ball = raw[raw["Object"] == "ball"].copy()

    home_mot = read_refined_mot(HOME_REFINED_MOT)
    away_mot = read_refined_mot(AWAY_REFINED_MOT)

    home_map, home_unmatched = map_refined_ids(players, home_mot, HOME_TEAM_ID)
    away_map, away_unmatched = map_refined_ids(players, away_mot, AWAY_TEAM_ID)

    players["Refined ID"] = pd.NA
    for idx, refined_id in {**home_map, **away_map}.items():
        players.loc[idx, "Refined ID"] = refined_id

    players = normalize_coordinates(players)
    ball = normalize_coordinates(ball)

    # The root CSV is the source of truth for the ball. Missing detections are
    # represented as NaN so event generation does not treat 0,0 as a real touch.
    ball.loc[(ball["X_Pitch"] == 0) & (ball["Y_Pitch"] == 0), ["x", "y"]] = math.nan

    frames = sorted(raw["Frame"].unique())

    home_rows, home_ids = make_metrica_team(players, ball, frames, HOME_TEAM_ID, "Home")
    away_rows, away_ids = make_metrica_team(players, ball, frames, AWAY_TEAM_ID, "Away")

    home_path = os.path.join(OUTDIR, f"Sample_Game_{GAME_ID}_RawTrackingData_Home_Team.csv")
    away_path = os.path.join(OUTDIR, f"Sample_Game_{GAME_ID}_RawTrackingData_Away_Team.csv")

    pd.DataFrame(home_rows).to_csv(home_path, index=False, header=False)
    pd.DataFrame(away_rows).to_csv(away_path, index=False, header=False)

    print("Saved:")
    print(home_path)
    print(away_path)
    print()
    print(f"Home refined player IDs: {home_ids}")
    print(f"Away refined player IDs: {away_ids}")
    print(f"Home rows matched: {len(home_map)} / {len(home_mot)}")
    print(f"Away rows matched: {len(away_map)} / {len(away_mot)}")
    print(f"Home unmatched MOT rows: {len(home_unmatched)}")
    print(f"Away unmatched MOT rows: {len(away_unmatched)}")

    missing_refined = players["Refined ID"].isna().sum()
    print(f"Original player rows without refined ID: {missing_refined}")


if __name__ == "__main__":
    main()
