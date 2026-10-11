"""Central configuration. Every tunable lives here."""
from dataclasses import dataclass, field

@dataclass
class TrackerConfig:
    high_conf: float = 0.5
    low_conf: float = 0.1
    min_hits_to_confirm: int = 3     # player_id minted only after this many matches
    max_speed_px_s: float = 900.0    # association gate: max plausible image-plane speed
    gate_height_factor: float = 0.6
    appearance_weight: float = 0.35
    reid_min_similarity: float = 0.80
    reid_ambiguity_margin: float = 0.08
    predict_ms: int = 1500           # predict (never confirm) position for this long
    lost_ms: int = 20000
    retire_ms: int = 120000
    ema_appearance: float = 0.9

@dataclass
class PipelineConfig:
    mode: str = "balanced"           # accuracy_first | balanced | high_throughput
    detector_weights: str = "weights/football-players.pt"  # you must supply this
    detector_imgsz: int = 1280
    cut_hist_threshold: float = 0.45
    ball_gap_ms: int = 800          # bridge ball gaps up to this long (flagged interp)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

MODES = {"accuracy_first": 0.0, "balanced": 1 / 15, "high_throughput": 1 / 8}  # min seconds between processed frames
