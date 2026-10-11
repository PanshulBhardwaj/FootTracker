# Metric definitions (v0.1)

Every metric returns `null` (shown as `—`) unless its evidence threshold is met. Zero is shown only when verified.

| Metric | Definition | Status in v0.1 |
|---|---|---|
| Possession | Share of *controlled* samples (ball within control radius of one team's player, calibrated). Loose-ball samples are excluded. Needs >= 30 controlled samples. | `metrics.possession` implemented; **event/ownership detector not yet wired** |
| Territory | Share of mapped positions/ball in each team's attacking half; needs attacking direction | Not implemented (returns null) |
| xThreat | Needs a trained/validated xT grid (e.g. Karun Singh's) applied to ball transitions | Not implemented |
| Danger entries | Entries of the ball into the zone in front of goal (define polygon in config) | Not implemented |
| Pass accuracy | completed / (completed + failed); `null` if denominator is 0 | Implemented, awaits verified pass events |
| Pressure index | Opponent proximity to ball carrier, closing speed, duration | Not implemented |
| Progressive passes | Completed pass that moves the ball >= 25% closer to goal | Not implemented |
| Ball recoveries | Controlled possession gained after opposition control or loose ball | Not implemented |
| Pitch control (UI) | Nearest-player split on mapped positions of the current frame. A proxy, not a velocity/probability model | Implemented in frontend |
| Heatmap | Gaussian-smoothed occupancy of confirmed, calibrated observations only | Implemented |

## Pass detection rules (v0.1, frontend/index.html)
- **Ball owner per frame:** nearest confirmed player whose feet are within 0.45 player-heights of the ball and the ball is below 60% of the box height. If an opponent is within 1.35x that distance, the ball is contested and has no owner. Ball detections below 0.25 confidence are ignored.
- **Possession segment:** same player owning the ball for >= 160 ms (or 2+ frames); gaps up to 400 ms are tolerated.
- **Pass:** two consecutive segments by different players within 3 s, with the ball moving >= 1.2 player-heights between them (shorter = touch/duel, not a pass).
  - same team, gap <= 2.5 s: `completed`; same team, longer gap: `uncertain` (excluded from accuracy)
  - different team: `failed` (lost to opponent). Out-of-play and deflections are not detected.
- **Possession %:** time-weighted share of frames with a clear owner; loose/contested/no-ball time is excluded. Needs >= 3 s of ownership.
- **Pass accuracy:** completed / (completed + failed) per team; unavailable if none.
- **Ball recoveries:** a team's player takes the ball <= 5 s after an opposing player had it (includes interceptions and tackles).
- **Progressive pass:** completed pass ending >= 25% and >= 5 m closer to the goal being attacked. Needs pitch calibration and the "Team A attacks" setting.
Known failure modes: tracker ID switches create false passes; ball occlusion creates missed passes; heading/aerial balls are weak.
- **Pressure index / pressing chart:** while a team has a clear ball owner, count opponents within 2.5 player-heights (image space) of the carrier; credited to the pressing team. Card = average count; chart = average per 5 s window, gaps left blank. Needs >= 8 samples per team. A proxy: not closing speed, not metres.
- **Ball in flight:** a ball moving faster than 5 player-heights/second (camera motion removed) cannot be 'owned', so fast passes are not mistaken for touches by bystanders.
- **Interpolated ball:** if the ball at either end of a pass is mostly interpolated (gap-filled, not detected), the pass is marked `uncertain` and excluded from accuracy.
- **Pass map:** arrows run from passer to receiver (solid = completed, orange = lost, dashed grey = uncertain). The same passes are drawn as arrows on the video for 1.5 s after they happen. `Passes CSV` exports them.
