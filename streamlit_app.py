from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INFO_FILE = Path("camera_info.csv")


st.set_page_config(
    page_title="Bridge Camera Monitor",
    page_icon="🌉",
    layout="wide",
)


def find_cameras(repository_path: Path) -> list[Path]:
    """Return direct subfolders of the repository, sorted by name."""
    if not repository_path.exists() or not repository_path.is_dir():
        return []

    return sorted(
        [p for p in repository_path.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )


def load_camera_info(camera_names: list[str]) -> pd.DataFrame:
    """Load manually maintained camera info, creating missing rows if needed."""
    if INFO_FILE.exists():
        df = pd.read_csv(INFO_FILE)
    else:
        df = pd.DataFrame(columns=["camera", "Debris", "Water", "bridge", "status"])

    required_columns = ["camera", "Debris", "Water", "bridge", "status"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    existing = set(df["camera"].astype(str))
    missing = [name for name in camera_names if name not in existing]

    if missing:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "camera": missing,
                        "Debris": ["" for _ in missing],
                        "Water": ["" for _ in missing],
                        "bridge": ["" for _ in missing],
                        "status": ["Unknown" for _ in missing],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Keep only current repository cameras, in repository order.
    df = df[df["camera"].isin(camera_names)].copy()
    df["camera"] = pd.Categorical(df["camera"], categories=camera_names, ordered=True)
    df = df.sort_values("camera").reset_index(drop=True)
    df["camera"] = df["camera"].astype(str)

    return df


def save_camera_info(df: pd.DataFrame) -> None:
    df.to_csv(INFO_FILE, index=False)


def camera_summary(camera_path: Path) -> dict:
    image_files = [
        p for p in camera_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    latest_file = max(image_files, key=lambda p: p.stat().st_mtime, default=None)

    return {
        "camera": camera_path.name,
        "path": str(camera_path),
        "image_count": len(image_files),
        "latest_image": latest_file.name if latest_file else "",
        "latest_modified": (
            datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if latest_file
            else ""
        ),
    }


def show_overview(camera_paths: list[Path], debris_dict, activity_dict, water_dict) -> None:
    st.title("🌉 Bridge Camera Monitor")

    camera_names = [p.name for p in camera_paths]

    if not camera_paths:
        st.warning("No camera folders found in the selected repository.")
        return

    summaries = pd.DataFrame([camera_summary(path) for path in camera_paths])
    info = load_camera_info(camera_names)

    overview = summaries.merge(info, on="camera", how="left")
    overview["Debris"] = overview["camera"].map(debris_dict)
    overview["Debris"] = overview["camera"].map(debris_dict).fillna(overview["Debris"])
    overview["status"] = overview["camera"].map(activity_dict)
    overview["status"] = overview["camera"].map(activity_dict).fillna(overview["status"])
    overview["Water"] = overview["camera"].map(water_dict)
    overview["Water"] = overview["camera"].map(water_dict).fillna(overview["Water"])
    print(overview)
    st.subheader("Cameras")

    edited = st.data_editor(
        overview,
        use_container_width=True,
        hide_index=True,
        disabled=["camera", "path", "image_count", "latest_image", "latest_modified", "Debris", "status", "Water"],
        column_order=[
            "camera",
            "bridge",
            "status",
            "Debris",
            "Water",
            #"image_count",
            #"latest_image",
            "latest_modified",
            #"path",
        ],
        column_config={
            "camera": st.column_config.TextColumn("Camera"),
            "bridge": st.column_config.TextColumn("Bridge"),
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["Unknown", "Active", "Inactive", "Needs review"],
            ),
            "Debris": st.column_config.TextColumn("Debris presence"),
            "Water": st.column_config.TextColumn("Water level"),
            #"image_count": st.column_config.NumberColumn("Images"),
            #"latest_image": st.column_config.TextColumn("Latest image"),
            "latest_modified": st.column_config.TextColumn("Latest modified"),
            #"path": st.column_config.TextColumn("Folder path"),
        },
    )

    if st.button("Save camera info"):
        info_columns = ["camera", "Debris", "bridge", "status"]
        save_camera_info(edited[info_columns])
        st.success("Camera info saved.")


def show_camera_page(camera_path: Path, debris_dict: dict = {}, activity_dict: dict = {}, water_dict: dict = {}) -> None:
    print('camera_path', camera_path)
    st.title(camera_path.name)

    summary = camera_summary(camera_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest modified", summary["latest_modified"] or "N/A")

    info_df = load_camera_info([camera_path.name])
    if not info_df.empty:
        row = info_df.iloc[0]

        activity_status = "—"
        if camera_path.name in activity_dict:
            activity_status = activity_dict[camera_path.name]
        debris_status = "—"
        if camera_path.name in debris_dict:
            debris_status = debris_dict[camera_path.name]
        water_status = "—"
        print(water_dict)
        if camera_path.name in water_dict:
            water_status = water_dict[camera_path.name]
        st.subheader("Camera info")
        st.write(f"**Bridge:** {row.get('bridge', '') or '—'}")
        #st.write(f"**Status:** {row.get('status', '') or '—'}")
        st.write(f"**Status:** {activity_status}")
        st.write(f"**Debris presence:** {debris_status}")
        st.write(f"**Water level:** {water_status}")

    st.subheader("Browse images by time")

    image_files = sorted(
        [
            p for p in camera_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not image_files:
        st.info("No image files found for this camera yet.")
        return

    image_index = st.slider(
        "Image position, sorted by most recent first",
        min_value=1,
        max_value=len(image_files),
        value=1,
        step=1,
        help="1 is the most recent image. 2 is the second most recent image, and so on.",
    )

    selected_image = image_files[image_index - 1]
    selected_timestamp = datetime.fromtimestamp(
        selected_image.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    left, center, right = st.columns([1, 3, 1])

    image_col, info_col = st.columns([3, 1])

    with image_col:
        st.image(
            str(selected_image),
            caption=f"{selected_image.name} — {selected_timestamp}",
            use_container_width=True,
        )

    with info_col:
        #st.write("**Image:**", selected_image.name)
        st.write("**Timestamp:**", selected_timestamp)


def process_debris_csv(csv_path: Path, n: int = 3):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return "No data", "Inactive"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        return "No data", "Inactive"
    m = df.nlargest(n, "date")["pixel_count"].mean()

    max_date = max(df["date"])
    status = "Active"
    print(csv_path, datetime.date(max_date))
    if datetime.today().date() - datetime.date(max_date) > timedelta(days=6):
        print('NIQUE')
        status = "Inactive"
    if m > 10000:
        return "High", status
    elif m > 5000:
        return "Medium", status
    elif m > 500:
        return "Low", status
    else:
        return "Clear", status

def process_water_csv(csv_path: Path, qtl_low: float = 0.05, qtl_high: float = 0.95, n: int = 3):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return "No data"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        return "No data"
    m_low = df['pixel_count'].quantile(qtl_low)
    m_high = df['pixel_count'].quantile(qtl_high)
    m_cur = df.nlargest(n, "date")["pixel_count"].mean()
    if m_cur < m_low:
        return "Low"
    elif m_cur > m_high:
        return "High"
    else:
        return "Normal"

with st.sidebar:
    st.header("Repository")

    repo_input = st.text_input(
        "Camera repository path",
        value="images/",
        help="Path to the folder containing one subfolder per camera.",
    )
    
    debris_log_input = st.text_input(
        "Debris CSVs repository",
        value="debris_csv/",
        help="Path to the folder containing the CSVs generated by the debris detection algorithm.",
    )

    water_log_input = st.text_input(
        "Water CSVs repository",
        value="water_csv/",
        help="Path to the folder containing the CSVs generated by the water detection algorithm.",
    )
    
    H_debris = {}
    H_activity = {}
    if os.path.exists(debris_log_input):
        for filename in os.listdir(debris_log_input):
            cam_name = filename.rstrip('.csv')
            (debris_status, activity_status) = process_debris_csv(Path(os.path.join(debris_log_input, filename)), n=3)
            H_debris[cam_name] = debris_status
            H_activity[cam_name] = activity_status

    H_water = {}
    if os.path.exists(water_log_input):
        for filename in os.listdir(water_log_input):
            cam_name = filename.rstrip('.csv')
            H_water[cam_name] = process_water_csv(Path(os.path.join(water_log_input, filename)), qtl_low=0.05, qtl_high=0.95, n=3)

    repository_path = Path(repo_input).expanduser().resolve()
    camera_paths = find_cameras(repository_path)
    camera_names = [p.name for p in camera_paths]

    st.divider()
    st.header("Pages")

    page_options = ["Overview"] + camera_names
    selected_page = st.radio("Select page", page_options, label_visibility="collapsed")


if selected_page == "Overview":
    show_overview(camera_paths, H_debris, H_activity, H_water)
else:
    selected_camera_path = repository_path / selected_page
    show_camera_page(selected_camera_path, debris_dict=H_debris, activity_dict=H_activity, water_dict=H_water)
