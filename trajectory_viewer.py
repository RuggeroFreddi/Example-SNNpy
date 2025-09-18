from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Config & paths
# ----------------------------

DATA_PATH = Path("dati/retina_gesture_dataset_32x32_with_random.npz")
LABEL_NAMES: List[str] = [
    "right→left",
    "left→right",
    "top→bottom",
    "bottom→top",
    "clockwise",
    "counterclockwise",
    "random",
]
INTERVAL_MS = 80  # frame interval for animation


# ----------------------------
# Data I/O
# ----------------------------

def load_dataset(path: Path = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load (N, T, H, W) data and (N,) label array from NPZ."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path) as npz:
        return npz["data"], npz["labels"]


def build_label_index(labels: np.ndarray, n_classes: int) -> Dict[int, np.ndarray]:
    """Map label_id -> array of sample indices."""
    return {label_id: np.where(labels == label_id)[0] for label_id in range(n_classes)}


# ----------------------------
# Animation
# ----------------------------

def animate_sample(sample: np.ndarray, title: str, interval_ms: int = INTERVAL_MS) -> None:
    """
    Animate all frames of a single sample (T, H, W).
    Instructions are shown on screen; key handling is attached by the caller.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sample[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{title} • t=0")
    ax.axis("off")

    # On-screen instructions
    fig.text(
        0.5, 0.02,
        "Press any key for next • Esc to exit",
        ha="center", va="bottom", fontsize=9
    )

    def update(frame_idx: int):
        im.set_data(sample[frame_idx])
        ax.set_title(f"{title} • t={frame_idx}")
        return (im,)

    FuncAnimation(
        fig,
        update,
        frames=sample.shape[0],
        interval=interval_ms,
        blit=True,
        repeat=True,
        cache_frame_data=False,
    )
    plt.show()


def browse_by_label(
    data: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    interval_ms: int = INTERVAL_MS,
) -> None:
    """
    Loop over gesture classes and show one random sample per class.
    - Any key: close current window and move to the next
    - Esc: close and exit
    Repeats indefinitely until Esc is pressed or the window is closed on Esc.
    """
    rng = np.random.default_rng()
    label_to_indices = build_label_index(labels, len(label_names))

    print("Dataset shape:", data.shape)  # (N, T, H, W)
    print("Available gesture classes:", len(label_names))

    exit_flag = False
    try:
        while not exit_flag:
            for label_id, label_name in enumerate(label_names):
                indices = label_to_indices.get(label_id, np.array([], dtype=int))
                if indices.size == 0:
                    print(f"⚠️  No samples for label {label_id} ({label_name})")
                    continue

                sample_idx = int(rng.choice(indices))
                sample = data[sample_idx]  # shape: (T, H, W)
                title = f"Sample #{sample_idx} — Gesture: {label_name}"
                print(f"\n▶️  {title}")

                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(sample[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{title} • t=0")
                ax.axis("off")

                fig.text(
                    0.5, 0.02,
                    "Press any key for next • Esc to exit",
                    ha="center", va="bottom", fontsize=9
                )

                def update(frame_idx: int):
                    im.set_data(sample[frame_idx])
                    ax.set_title(f"{title} • t={frame_idx}")
                    return (im,)

                anim = FuncAnimation(
                    fig,
                    update,
                    frames=sample.shape[0],
                    interval=interval_ms,
                    blit=True,
                    repeat=True,
                    cache_frame_data=False,
                )

                # Key handling: any key -> next; Esc -> exit
                def on_key(event):
                    nonlocal exit_flag
                    if event.key == "escape":
                        exit_flag = True
                    plt.close(fig)  # proceed (or exit if Esc)

                cid = fig.canvas.mpl_connect("key_press_event", on_key)
                plt.show()
                fig.canvas.mpl_disconnect(cid)

                if exit_flag:
                    break
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    print("✔ Animation browsing finished.")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    data, labels = load_dataset()
    browse_by_label(data, labels, LABEL_NAMES, interval_ms=INTERVAL_MS)


if __name__ == "__main__":
    main()
