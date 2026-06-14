import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "data/gp_data_normal.csv"
OUTPUT_FILE = "data/gps_data_spoofed3-a.csv"
ATTACK_LOG_FILE = "outputs/attack_windows_report.csv"
PLOTS_DIR = "outputs/spoofing_plots"

RANDOM_SEED = 42
TARGET_SPOOF_RATIO = 0.28     # حوالي 28% spoofed
NUM_ATTACK_ROUNDS = 10      # 10 attack windows so each attack type appears ~2x
MIN_GAP_BETWEEN_ATTACKS = 180 # seconds/records تقريباً لأن الداتا كل ثانية
START_END_BUFFER = 240        # اتركي بداية ونهاية الداتا normal
NUM_SESSIONS = 3              # split data into N artificial sessions for generalization

# Human realistic limits based on your dataset
MAX_NORMAL_HUMAN_SPEED = 2.8  # m/s, most walking/running-like values stay below this
MAX_SHORT_TRANSITION_SPEED = 3.6  # base cap; randomized per round in recalculate_motion_for_window

# ============================================================
# GPS HELPERS
# ============================================================
EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def destination_point(lat, lon, distance_m, bearing_degrees):
    """Move from lat/lon by distance_m at bearing_degrees."""
    bearing = math.radians(bearing_degrees)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    ang_dist = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist)
        + math.cos(lat1) * math.sin(ang_dist) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def smoothstep(x):
    """Smooth transition from 0 to 1."""
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ============================================================
# DATA PREPARATION
# ============================================================
def validate_columns(df):
    required = [
        "session_id", "gps_date", "gps_time", "latitude", "longitude",
        "velocity", "course", "satellites_in_view", "satellites_used", "hdop", "label"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + str(missing) + "\n"
            "Available columns: " + str(list(df.columns))
        )


def parse_datetime(df):
    dt = pd.to_datetime(
        df["gps_date"].astype(str) + " " + df["gps_time"].astype(str),
        errors="coerce"
    )
    return dt


def fill_missing_course(df):
    """
    Fill missing course using coordinate bearing.
    If movement is too small, keep previous valid course; if none, use 0.
    """
    df = df.copy()
    course = df["course"].copy()

    for i in range(len(df)):
        if pd.notna(course.iloc[i]):
            continue

        if i > 0:
            dist = haversine_m(
                df.loc[df.index[i - 1], "latitude"], df.loc[df.index[i - 1], "longitude"],
                df.loc[df.index[i], "latitude"], df.loc[df.index[i], "longitude"]
            )
            if dist > 0.5:
                course.iloc[i] = bearing_deg(
                    df.loc[df.index[i - 1], "latitude"], df.loc[df.index[i - 1], "longitude"],
                    df.loc[df.index[i], "latitude"], df.loc[df.index[i], "longitude"]
                )
            else:
                course.iloc[i] = course.iloc[i - 1] if pd.notna(course.iloc[i - 1]) else 0.0
        else:
            course.iloc[i] = 0.0

    df["course"] = course.fillna(method="ffill").fillna(0.0)
    return df


# ============================================================
# ATTACK WINDOW SELECTION
# ============================================================
def make_attack_durations(total_records, target_ratio, num_rounds, rng):
    target_total = int(round(total_records * target_ratio))

    # For your 7423 rows, this naturally creates about 5 rounds of 6-8 minutes.
    base = target_total // num_rounds
    durations = []
    for _ in range(num_rounds):
        jitter = rng.integers(-70, 71)
        durations.append(max(240, int(base + jitter)))

    # Adjust sum exactly to target_total
    diff = target_total - sum(durations)
    i = 0
    while diff != 0:
        step = 1 if diff > 0 else -1
        if durations[i % num_rounds] + step >= 180:
            durations[i % num_rounds] += step
            diff -= step
        i += 1

    return durations


def choose_attack_windows(n, durations):
    """
    Choose non-overlapping windows in chronological order.
    They are separated by normal periods and do not touch the start/end of the dataset.
    """
    total_attack = sum(durations)
    available_normal = n - total_attack - (2 * START_END_BUFFER)
    if available_normal <= (len(durations) + 1) * 30:
        raise ValueError("Dataset is too small for these attack durations/buffers.")

    gap = available_normal // (len(durations) + 1)
    windows = []
    cursor = START_END_BUFFER + gap

    for duration in durations:
        start = int(cursor)
        end = int(start + duration - 1)
        windows.append((start, end, duration))
        cursor = end + 1 + gap

    return windows


def split_phases(start, end):
    length = end - start + 1
    capture_len = max(20, int(length * 0.10))
    takeover_len = max(30, int(length * 0.15))
    release_len = max(30, int(length * 0.15))
    stable_len = length - capture_len - takeover_len - release_len
    if stable_len < 60:
        stable_len = max(30, stable_len)

    capture = (start, start + capture_len - 1)
    takeover = (capture[1] + 1, capture[1] + takeover_len)
    stable = (takeover[1] + 1, takeover[1] + stable_len)
    release = (stable[1] + 1, end)
    return {
        "capture": capture,
        "takeover": takeover,
        "stable": stable,
        "release": release,
    }


# ============================================================
# SIGNAL QUALITY MODIFICATION
# ============================================================
def modify_signal_quality(df, idxs, phase, rng):
    """
    Modify satellites and hdop slightly.
    The goal: not too obvious, but enough cross-feature inconsistency to detect.
    """
    if phase == "capture":
        drop_used_range = (0, 1)
        hdop_mult_range = (1.05, 1.25)
        view_drop_prob = 0.25
    elif phase == "takeover":
        drop_used_range = (1, 3)
        hdop_mult_range = (1.25, 2.20)
        view_drop_prob = 0.55
    elif phase == "stable":
        drop_used_range = (0, 2)
        hdop_mult_range = (1.10, 1.70)
        view_drop_prob = 0.35
    else:  # release
        drop_used_range = (1, 3)
        hdop_mult_range = (1.20, 2.00)
        view_drop_prob = 0.50

    for idx in idxs:
        original_view = int(df.at[idx, "satellites_in_view"])
        original_used = int(df.at[idx, "satellites_used"])
        original_hdop = float(df.at[idx, "hdop"])

        view = original_view
        if rng.random() < view_drop_prob:
            view -= int(rng.integers(0, 3))
        view = int(np.clip(view, 7, 14))

        used_drop = int(rng.integers(drop_used_range[0], drop_used_range[1] + 1))
        used = original_used - used_drop
        used = int(np.clip(used, 4, min(view, 10)))

        # Sometimes keep signal nearly normal so attack is not too easy.
        if phase == "stable" and rng.random() < 0.30:
            used = int(np.clip(original_used, 4, min(view, 10)))

        hdop = original_hdop * rng.uniform(*hdop_mult_range) + rng.uniform(0.00, 0.18)
        hdop = float(np.clip(hdop, 0.80, 4.00))

        df.at[idx, "satellites_in_view"] = view
        df.at[idx, "satellites_used"] = used
        df.at[idx, "hdop"] = round(hdop, 2)


# ============================================================
# ATTACK GENERATORS
# ============================================================
def apply_freeze_attack(df, start, end, phases, round_id, rng):
    """Fake location stays almost fixed near the attack start point with small GPS jitter."""
    lat0 = df.at[start, "latitude"]
    lon0 = df.at[start, "longitude"]

    # Small offset makes it look like an allowed/fake point, not exact copy.
    offset_m = rng.uniform(8, 25)
    offset_bearing = rng.uniform(0, 360)
    fake_lat, fake_lon = destination_point(lat0, lon0, offset_m, offset_bearing)

    fake_points = {}
    for idx in range(start, end + 1):
        if phases["capture"][0] <= idx <= phases["capture"][1]:
            # Keep original coordinates during capture, only signal quality changes.
            fake_points[idx] = (df.at[idx, "latitude"], df.at[idx, "longitude"])
        elif phases["takeover"][0] <= idx <= phases["takeover"][1]:
            denom = max(1, phases["takeover"][1] - phases["takeover"][0])
            t = smoothstep((idx - phases["takeover"][0]) / denom)
            real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
            jitter_m = rng.uniform(0, 1.5)
            jlat, jlon = destination_point(fake_lat, fake_lon, jitter_m, rng.uniform(0, 360))
            fake_points[idx] = ((1 - t) * real_lat + t * jlat, (1 - t) * real_lon + t * jlon)
        elif phases["stable"][0] <= idx <= phases["stable"][1]:
            jitter_m = rng.uniform(0.3, 2.5)
            fake_points[idx] = destination_point(fake_lat, fake_lon, jitter_m, rng.uniform(0, 360))
        else:  # release
            denom = max(1, phases["release"][1] - phases["release"][0])
            t = smoothstep((idx - phases["release"][0]) / denom)
            real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
            jitter_m = rng.uniform(0, 1.5)
            jlat, jlon = destination_point(fake_lat, fake_lon, jitter_m, rng.uniform(0, 360))
            fake_points[idx] = ((1 - t) * jlat + t * real_lat, (1 - t) * jlon + t * real_lon)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "freeze")


def find_replay_source(df, start, stable_len):
    """Find an earlier normal segment with a close starting location if possible."""
    min_source_end = 60
    max_start = start - stable_len - 120
    if max_start <= min_source_end:
        return None

    current_lat = df.at[start, "latitude"]
    current_lon = df.at[start, "longitude"]

    # Check candidates every 10 records to keep it fast.
    candidates = list(range(min_source_end, max_start, 10))
    if not candidates:
        return None

    best_candidate = None
    best_dist = float("inf")
    for cand in candidates:
        # Avoid using already spoofed rows as replay source.
        if df.loc[cand:cand + stable_len - 1, "label"].sum() != 0:
            continue
        d = haversine_m(current_lat, current_lon, df.at[cand, "latitude"], df.at[cand, "longitude"])
        if d < best_dist:
            best_dist = d
            best_candidate = cand

    return best_candidate


def apply_replay_attack(df, start, end, phases, round_id, rng):
    """Replay an older real segment, with current timestamps and slight signal changes."""
    stable_start, stable_end = phases["stable"]
    stable_len = stable_end - stable_start + 1
    source_start = find_replay_source(df, start, stable_len)

    if source_start is None:
        # Fallback: behave like a gradual drag if there is no earlier segment.
        apply_gradual_drag_attack(df, start, end, phases, round_id, rng, attack_name="replay_fallback_drag")
        return

    source_idxs = list(range(source_start, source_start + stable_len))
    replay_lats = df.loc[source_idxs, "latitude"].to_numpy()
    replay_lons = df.loc[source_idxs, "longitude"].to_numpy()

    fake_points = {}

    # Capture: original location with signal disturbance only.
    for idx in range(phases["capture"][0], phases["capture"][1] + 1):
        fake_points[idx] = (df.at[idx, "latitude"], df.at[idx, "longitude"])

    # Takeover: smooth movement from real location to first replay point.
    target_lat, target_lon = replay_lats[0], replay_lons[0]
    for idx in range(phases["takeover"][0], phases["takeover"][1] + 1):
        denom = max(1, phases["takeover"][1] - phases["takeover"][0])
        t = smoothstep((idx - phases["takeover"][0]) / denom)
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
        fake_points[idx] = ((1 - t) * real_lat + t * target_lat, (1 - t) * real_lon + t * target_lon)

    # Stable replay: old coordinates appear at current time.
    for k, idx in enumerate(range(stable_start, stable_end + 1)):
        fake_points[idx] = (replay_lats[k], replay_lons[k])

    # Release: return from last replay point to real current location.
    last_lat, last_lon = replay_lats[-1], replay_lons[-1]
    for idx in range(phases["release"][0], phases["release"][1] + 1):
        denom = max(1, phases["release"][1] - phases["release"][0])
        t = smoothstep((idx - phases["release"][0]) / denom)
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
        fake_points[idx] = ((1 - t) * last_lat + t * real_lat, (1 - t) * last_lon + t * real_lon)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "replay")
    df.loc[start:end, "replay_source_start_index"] = source_start


def apply_gradual_drag_attack(df, start, end, phases, round_id, rng, attack_name="gradual_drag"):
    """Reported location is slowly dragged away from the true path and then returned."""
    final_offset_m = rng.uniform(50, 150)  # larger range for realistic geofence evasion
    offset_bearing = rng.uniform(0, 360)

    fake_points = {}
    for idx in range(start, end + 1):
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]

        if phases["capture"][0] <= idx <= phases["capture"][1]:
            offset = 0.0
        elif phases["takeover"][0] <= idx <= phases["takeover"][1]:
            denom = max(1, phases["takeover"][1] - phases["takeover"][0])
            t = smoothstep((idx - phases["takeover"][0]) / denom)
            offset = final_offset_m * t
        elif phases["stable"][0] <= idx <= phases["stable"][1]:
            # Keep offset but add small natural variation.
            offset = final_offset_m + rng.normal(0, 2.0)
        else:
            denom = max(1, phases["release"][1] - phases["release"][0])
            t = smoothstep((idx - phases["release"][0]) / denom)
            offset = final_offset_m * (1 - t)

        lat2, lon2 = destination_point(real_lat, real_lon, max(0.0, offset), offset_bearing)
        fake_points[idx] = (lat2, lon2)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, attack_name)


def apply_fake_walking_attack(df, start, end, phases, round_id, rng):
    """Create a plausible fake walking path around a nearby fake point."""
    start_lat = df.at[start, "latitude"]
    start_lon = df.at[start, "longitude"]
    offset_m = rng.uniform(20, 60)
    fake_lat, fake_lon = destination_point(start_lat, start_lon, offset_m, rng.uniform(0, 360))

    fake_points = {}

    # Generate stable fake walking trajectory first.
    stable_start, stable_end = phases["stable"]
    current_lat, current_lon = fake_lat, fake_lon
    current_bearing = rng.uniform(0, 360)
    stable_points = []
    for _ in range(stable_start, stable_end + 1):
        speed = max(0.0, rng.normal(1.05, 0.25))  # human walking speed
        current_bearing = (current_bearing + rng.normal(0, 8)) % 360
        current_lat, current_lon = destination_point(current_lat, current_lon, speed, current_bearing)
        stable_points.append((current_lat, current_lon))

    # Capture original.
    for idx in range(phases["capture"][0], phases["capture"][1] + 1):
        fake_points[idx] = (df.at[idx, "latitude"], df.at[idx, "longitude"])

    # Takeover to first fake walking point.
    target_lat, target_lon = stable_points[0]
    for idx in range(phases["takeover"][0], phases["takeover"][1] + 1):
        denom = max(1, phases["takeover"][1] - phases["takeover"][0])
        t = smoothstep((idx - phases["takeover"][0]) / denom)
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
        fake_points[idx] = ((1 - t) * real_lat + t * target_lat, (1 - t) * real_lon + t * target_lon)

    # Stable fake walking.
    for k, idx in enumerate(range(stable_start, stable_end + 1)):
        fake_points[idx] = stable_points[k]

    # Release back to real.
    last_lat, last_lon = stable_points[-1]
    for idx in range(phases["release"][0], phases["release"][1] + 1):
        denom = max(1, phases["release"][1] - phases["release"][0])
        t = smoothstep((idx - phases["release"][0]) / denom)
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]
        fake_points[idx] = ((1 - t) * last_lat + t * real_lat, (1 - t) * last_lon + t * real_lon)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "fake_walking")


def apply_small_jump_attack(df, start, end, phases, round_id, rng):
    """A moderate jump followed by stable spoofing and release. Used only in one round."""
    jump_m = rng.uniform(55, 110)
    jump_bearing = rng.uniform(0, 360)
    base_lat, base_lon = destination_point(df.at[start, "latitude"], df.at[start, "longitude"], jump_m, jump_bearing)

    fake_points = {}
    for idx in range(start, end + 1):
        real_lat, real_lon = df.at[idx, "latitude"], df.at[idx, "longitude"]

        if phases["capture"][0] <= idx <= phases["capture"][1]:
            fake_points[idx] = (real_lat, real_lon)
        elif phases["takeover"][0] <= idx <= phases["takeover"][1]:
            # Faster than gradual drag, but still smoothed to avoid impossible huge speed every row.
            denom = max(1, phases["takeover"][1] - phases["takeover"][0])
            t = smoothstep((idx - phases["takeover"][0]) / denom)
            fake_points[idx] = ((1 - t) * real_lat + t * base_lat, (1 - t) * real_lon + t * base_lon)
        elif phases["stable"][0] <= idx <= phases["stable"][1]:
            # Small motion around jumped point.
            jitter_m = rng.uniform(0.5, 3.0)
            fake_points[idx] = destination_point(base_lat, base_lon, jitter_m, rng.uniform(0, 360))
        else:
            denom = max(1, phases["release"][1] - phases["release"][0])
            t = smoothstep((idx - phases["release"][0]) / denom)
            fake_points[idx] = ((1 - t) * base_lat + t * real_lat, (1 - t) * base_lon + t * real_lon)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "small_jump_recovery")


# ============================================================
# WRITE / RECALCULATE FEATURES
# ============================================================
def write_fake_points(df, fake_points):
    for idx, (lat, lon) in fake_points.items():
        df.at[idx, "latitude"] = float(lat)
        df.at[idx, "longitude"] = float(lon)


def set_attack_metadata(df, start, end, round_id, attack_type):
    df.loc[start:end, "label"] = 1
    df.loc[start:end, "is_generated_spoof"] = 1
    df.loc[start:end, "attack_round_id"] = round_id
    df.loc[start:end, "attack_type"] = attack_type


def assign_phase_metadata(df, phases):
    for phase_name, (p_start, p_end) in phases.items():
        df.loc[p_start:p_end, "attack_phase"] = phase_name


def recalculate_motion_for_window(df, start, end, rng):
    """
    Recalculate velocity and course for the modified trajectory.
    Adds small reporting noise so it is not mathematically perfect.
    """
    for idx in range(start, end + 1):
        if idx == 0:
            continue

        lat1, lon1 = df.at[idx - 1, "latitude"], df.at[idx - 1, "longitude"]
        lat2, lon2 = df.at[idx, "latitude"], df.at[idx, "longitude"]
        dist = haversine_m(lat1, lon1, lat2, lon2)

        dt = df.at[idx, "time_delta_sec"] if "time_delta_sec" in df.columns else 1.0
        if pd.isna(dt) or dt <= 0:
            dt = 1.0

        speed = dist / dt

        # Keep most spoofed motion human-like. Allow short transitions to be slightly higher.
        phase = df.at[idx, "attack_phase"]
        # Jitter the transition cap per call so no two rounds share the exact same ceiling.
        transition_cap = MAX_SHORT_TRANSITION_SPEED * rng.uniform(0.88, 1.12)
        max_speed = transition_cap if phase in ["takeover", "release"] else MAX_NORMAL_HUMAN_SPEED
        reported_speed = speed * rng.uniform(0.88, 1.12) + rng.normal(0, 0.04)
        # Floor at 0.05 so GPS never shows exact zero (real receivers always show small noise)
        reported_speed = float(np.clip(reported_speed, 0.05, max_speed))

        if dist > 0.6:
            crs = bearing_deg(lat1, lon1, lat2, lon2)
            crs = (crs + rng.normal(0, 4.0)) % 360
        else:
            # If almost stationary, course is naturally unstable/less meaningful.
            prev_course = df.at[idx - 1, "course"] if pd.notna(df.at[idx - 1, "course"]) else 0.0
            crs = (prev_course + rng.normal(0, 6.0)) % 360
            if reported_speed < 0.10 and rng.random() < 0.40:
                crs = 0.0

        df.at[idx, "velocity"] = round(reported_speed, 3)
        df.at[idx, "course"] = round(crs, 2)


def compute_time_deltas(df):
    dt = parse_datetime(df)
    df["_datetime"] = dt
    df["time_delta_sec"] = dt.diff().dt.total_seconds()
    df["time_delta_sec"] = df["time_delta_sec"].fillna(1.0)
    df.loc[df["time_delta_sec"] <= 0, "time_delta_sec"] = 1.0
    return df


def ensure_session_id(df, num_sessions=NUM_SESSIONS):
    """
    Preserve real session_id if it already exists.
    If the input dataset has no usable session_id, create artificial sessions.

    Important:
    - Do NOT train the model on session_id.
    - Use session_id later only to split windows correctly and avoid mixing
      the last row of one recording with the first row of another recording.
    """
    df = df.copy()

    if "session_id" in df.columns:
        session_text = df["session_id"].astype(str).str.strip()
        has_real_session = df["session_id"].notna().any() and (session_text != "").any() and (session_text.str.lower() != "nan").any()

        if has_real_session:
            df["session_id"] = (
                df["session_id"]
                .ffill()
                .bfill()
                .fillna("unknown_session")
                .astype(str)
            )
            return df

    # Fallback only if session_id is missing or completely empty.
    n = len(df)
    safe_num_sessions = max(1, min(int(num_sessions), n))
    session_size = int(math.ceil(n / safe_num_sessions))

    df["session_id"] = [
        f"session_{min(i // session_size + 1, safe_num_sessions)}"
        for i in range(n)
    ]
    return df


# ============================================================
# PLOTS AND REPORTS
# ============================================================
def create_plots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    x = np.arange(len(df))

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["label"].to_numpy())
    plt.title("Label Timeline: Normal vs Spoofed Attack Windows")
    plt.xlabel("Record index")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "label_timeline.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["velocity"].to_numpy())
    plt.title("Velocity Over Time")
    plt.xlabel("Record index")
    plt.ylabel("Velocity (m/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "velocity_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["hdop"].to_numpy())
    plt.title("HDOP Over Time")
    plt.xlabel("Record index")
    plt.ylabel("HDOP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hdop_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["satellites_used"].to_numpy())
    plt.title("Satellites Used Over Time")
    plt.xlabel("Record index")
    plt.ylabel("Satellites used")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "satellites_used_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    normal = df[df["label"] == 0]
    spoofed = df[df["label"] == 1]
    plt.scatter(normal["longitude"], normal["latitude"], s=3, alpha=0.45, label="normal")
    plt.scatter(spoofed["longitude"], spoofed["latitude"], s=3, alpha=0.65, label="spoofed")
    plt.title("Trajectory: Normal and Spoofed Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_normal_spoofed.png"), dpi=200)
    plt.close()


def print_summary(df, attack_log):
    total = len(df)
    spoofed = int((df["label"] == 1).sum())
    normal = total - spoofed

    print("\n" + "=" * 70)
    print("REALISTIC HUMAN GPS SPOOFING DATASET GENERATED")
    print("=" * 70)
    print(f"Total records:   {total}")
    print(f"Normal records:  {normal} ({normal / total * 100:.2f}%)")
    print(f"Spoofed records: {spoofed} ({spoofed / total * 100:.2f}%)")
    print("\nAttack windows:")
    for item in attack_log:
        print(
            f"  Round {item['attack_round_id']}: "
            f"{item['attack_type']:<20} "
            f"start={item['start_index']:<5} end={item['end_index']:<5} "
            f"duration={item['duration_records']} records"
        )

    print("\nFeature ranges after generation:")
    for col in ["velocity", "satellites_in_view", "satellites_used", "hdop"]:
        print(
            f"  {col:<20} min={df[col].min():.3f}  "
            f"mean={df[col].mean():.3f}  max={df[col].max():.3f}"
        )
    print("=" * 70)


# ============================================================
# MAIN GENERATION PIPELINE
# ============================================================
def generate_spoofed_dataset(input_file=INPUT_FILE):
    rng = np.random.default_rng(RANDOM_SEED)

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Put the CSV file in the same folder as this script, or change INPUT_FILE."
        )

    df = pd.read_csv(input_file)
    validate_columns(df)

    # Keep original row order after sorting by time.
    df = df.copy().reset_index(drop=True)
    df = compute_time_deltas(df)
    df = fill_missing_course(df)

    # Preserve real session_id if it exists; otherwise create artificial sessions.
    df = ensure_session_id(df)

    # Ensure original dataset starts as normal.
    df["label"] = 0

    # Metadata columns for documentation only. Do NOT train the ML model on these columns.
    df["is_generated_spoof"] = 0
    df["attack_round_id"] = 0
    df["attack_type"] = "normal"
    df["attack_phase"] = "normal"
    df["replay_source_start_index"] = np.nan

    durations = make_attack_durations(len(df), TARGET_SPOOF_RATIO, NUM_ATTACK_ROUNDS, rng)
    windows = choose_attack_windows(len(df), durations)

    # Fixed balanced plan: not too easy, not too hard.
    attack_types = [
        "gradual_drag",
        "freeze",
        "replay",
        "fake_walking",
        "small_jump_recovery",
    ]

    # Guaranteed balanced attack plan:
    # With 10 rounds and 5 attack types, each type appears exactly 2 times.
    # The order is shuffled, so the pattern is not predictable.
    attack_plan = (attack_types * int(math.ceil(NUM_ATTACK_ROUNDS / len(attack_types))))[:NUM_ATTACK_ROUNDS]
    rng.shuffle(attack_plan)

    attack_log = []

    for round_i, (start, end, duration) in enumerate(windows, start=1):
        attack_type = attack_plan[round_i - 1]
        phases = split_phases(start, end)
        assign_phase_metadata(df, phases)

        if attack_type == "gradual_drag":
            apply_gradual_drag_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "freeze":
            apply_freeze_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "replay":
            apply_replay_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "fake_walking":
            apply_fake_walking_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "small_jump_recovery":
            apply_small_jump_attack(df, start, end, phases, round_i, rng)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Signal quality changes per phase.
        for phase_name, (p_start, p_end) in phases.items():
            modify_signal_quality(df, range(p_start, p_end + 1), phase_name, rng)

        # Recalculate velocity/course after coordinate modification.
        recalculate_motion_for_window(df, start, end, rng)

        attack_log.append({
            "attack_round_id": round_i,
            "attack_type": attack_type,
            "start_index": start,
            "end_index": end,
            "duration_records": duration,
            "capture_start": phases["capture"][0],
            "capture_end": phases["capture"][1],
            "takeover_start": phases["takeover"][0],
            "takeover_end": phases["takeover"][1],
            "stable_start": phases["stable"][0],
            "stable_end": phases["stable"][1],
            "release_start": phases["release"][0],
            "release_end": phases["release"][1],
        })

    # Final sanity checks.
    df["satellites_in_view"] = df["satellites_in_view"].round().astype(int)
    df["satellites_used"] = df["satellites_used"].round().astype(int)
    df["satellites_used"] = np.minimum(df["satellites_used"], df["satellites_in_view"])
    df["hdop"] = df["hdop"].round(2)
    df["velocity"] = df["velocity"].round(3)
    df["course"] = df["course"].round(2)

    # Remove internal datetime if you do not want it in the final dataset.
    # Keep time_delta_sec because it is useful later for detection feature engineering.
    if "_datetime" in df.columns:
        df = df.drop(columns=["_datetime"])

    # Drop internal generation metadata — not needed in the output CSV.
    cols_to_drop = [
        "is_generated_spoof", "attack_round_id", "attack_type",
        "attack_phase", "replay_source_start_index", "time_delta_sec"
    ]
    output_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Save outputs.
    output_df.to_csv(OUTPUT_FILE, index=False)
    attack_log_df = pd.DataFrame(attack_log)
    attack_log_df.to_csv(ATTACK_LOG_FILE, index=False)
    create_plots(df)
    print_summary(df, attack_log)

    print(f"\nSaved dataset: {OUTPUT_FILE}")
    print(f"Saved attack log: {ATTACK_LOG_FILE}")
    print(f"Saved plots folder: {PLOTS_DIR}/")
    print("\nImportant: For model training, do NOT use attack_round_id, attack_type, attack_phase, or is_generated_spoof as features.")
    print("Use them only for documentation and evaluation per attack type.")

    return df, attack_log_df


if __name__ == "__main__":
    generate_spoofed_dataset(INPUT_FILE)
