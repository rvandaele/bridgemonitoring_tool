import argparse
import base64
import sys, os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import imageio.v2 as imageio
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import numpy as np
import matplotlib.pyplot as plt

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify"
]

GMAIL_QUERY = 'newer_than:7d (has:attachment OR has:inline)'

POLL_SECONDS = 60

def trash_message(service, msg_id: str, user_id="me") -> None:
    service.users().messages().trash(userId=user_id, id=msg_id).execute()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract images from Gmail attachments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images"),
        help="Base directory where camera subfolders and images will be stored.",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default="credentials.json",
        help="Path to the Gmail OAuth credentials JSON file.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("masks/"),
        help="Path to the reolink masks used to hide personal data"
    )
    return parser.parse_args()


def gmail_service(credentials_json_path: str = "credentials.json"):
    creds = None
    token_path = "token.pickle"

    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_json_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def list_message_ids(service, user_id="me", query: str = "") -> List[str]:
    msg_ids = []
    page_token = None

    while True:
        kwargs = dict(userId=user_id, q=query, maxResults=500)
        if page_token:
            kwargs["pageToken"] = page_token

        resp = service.users().messages().list(**kwargs).execute()
        msgs = resp.get("messages", [])
        msg_ids.extend(m["id"] for m in msgs)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

        print(f"Fetched {len(msg_ids)} message IDs so far...")

    return msg_ids


def get_message_full(service, msg_id: str, user_id="me") -> Dict[str, Any]:
    return service.users().messages().get(
        userId=user_id,
        id=msg_id,
        format="full"
    ).execute()


def safe_filename(name: str) -> str:
    keep = " ._-"
    return "".join(c for c in name if c.isalnum() or c in keep).strip() or "image"


def download_attachment(service, msg_id: str, attachment_id: str, user_id="me") -> bytes:
    att = service.users().messages().attachments().get(
        userId=user_id,
        messageId=msg_id,
        id=attachment_id
    ).execute()
    data = att["data"]
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def walk_parts(part: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    stack = [part]
    while stack:
        p = stack.pop()
        out.append(p)
        for child in p.get("parts", []) or []:
            stack.append(child)
    return out


def extract_images_from_message(service, msg: Dict[str, Any], output_dir: Path, reolink_masks) -> int:
    msg_id = msg["id"]
    payload = msg.get("payload", {})
    parts = walk_parts(payload)

    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    subject = headers.get("subject", "no-subject")
    from_ = headers.get("from", "unknown-sender")
    stamp = msg.get("internalDate", "0")

    CAMERA_DIRS = {
        "Bickleigh":    output_dir / "reolink_0",
        "Halfpenny":    output_dir / "reolink_6",
        "Reolink DCC 1": output_dir / "reolink_1",
        "Reolink DCC 3": output_dir / "reolink_3",
        "Reolink DCC 4": output_dir / "reolink_4",
        "Reolink DCC 5": output_dir / "reolink_5",
        "Reolink DCC 6": output_dir / "reolink_6",
        "Reolink DCC 7": output_dir / "reolink_7",
        "Reolink DCC 8": output_dir / "reolink_8",
    }

    saved = 0
    for p in parts:
        mime = (p.get("mimeType") or "").lower()
        if not mime.startswith("image/"):
            continue

        body = p.get("body", {}) or {}
        attachment_id = body.get("attachmentId")
        if not attachment_id:
            data = body.get("data")
            if data:
                raw = base64.urlsafe_b64decode(data.encode("utf-8"))
            else:
                continue
        else:
            raw = download_attachment(service, msg_id, attachment_id)

        filename = p.get("filename") or f"{mime.replace('/', '_')}.bin"
        filename = safe_filename(filename)

        base = safe_filename(subject)[:60]
        sender = safe_filename(from_)[:40]
        out_name = f"{base}__{sender}__{stamp}__{saved}__{filename}"

        out_dir = next(
            (d for key, d in CAMERA_DIRS.items() if key in out_name),
            output_dir,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        cam = str(out_dir).split('/')[-1]

        with open(out_path, "wb") as f:
            f.write(raw)

        if cam in reolink_masks and np.max(reolink_masks[cam]) != -1:
            img = imageio.imread(out_path)
            (h, w, d) = img.shape
            (hm, wm) = reolink_masks[cam].shape
            if hm == h and wm == w:
                for i in range(3):
                    img[:, :, i] *= reolink_masks[cam]
            imageio.imwrite(out_path, img.astype('uint8'))
        else:
            print('no mask found for camera %s, writing unmasked' % cam)
        saved += 1

    return saved


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reolink_masks = {}
    for i in range(9):
        fullpath = os.path.join(args.mask_dir, 'reolink_%d.png' % i)
        if os.path.exists(fullpath):
            reolink_masks['reolink_%d' % i] = 1 - imageio.imread(fullpath)
        else:
            reolink_masks['reolink_%d' % i] = -1


    service = gmail_service(args.credentials)
    processed = set()

    while True:
        try:
            msg_ids = list_message_ids(service, query=GMAIL_QUERY)
            for msg_id in reversed(msg_ids):
                if msg_id in processed:
                    continue

                msg = get_message_full(service, msg_id)
                count = extract_images_from_message(service, msg, output_dir, reolink_masks)
                print(f"Processed {msg_id}: saved {count} images")
                if count > 0:
                    trash_message(service, msg_id)
                    print(f"Trashed {msg_id}")
                processed.add(msg_id)

        except HttpError as e:
            print("Gmail API error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
