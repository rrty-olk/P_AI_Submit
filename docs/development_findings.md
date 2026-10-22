# Development findings - 2026-10-22

## Confirmed observations

- Windows RTX 3070, 8 GiB nominal VRAM, was available through WSL 2 and an NVIDIA-enabled Docker container.
- Existing host distribution: Ubuntu 22.04.5 LTS. Application image: Ubuntu 24.04.5 LTS, ROS 2 Jazzy.
- The model's real official `system_one` call provides the three option scores in one forward pass.
- Initial verbose inspection descriptions produced all PASS predictions on the nine synthetic images.
  This was an unsuccessful inspection result (3/9 correct, only one route) and was not used for a submission video.
- A four-prompt comparison on three development images was performed with the same GPU model.
  Full results: `output/logs/prompt_experiments.json`.
- The concise intact/damaged/obscured wording distinguished the development intact and damaged images.
  The development obscured image still had PASS as its top answer, at confidence about 0.567;
  the pre-existing 0.60 threshold therefore directs it to RECHECK after stopping.
- A stricter audit discovered a backward wall-clock correction during ROS telemetry collection.
  It overlapped later motion with the earlier stop window in timestamp-based replay. The logger was
  changed to an epoch origin plus monotonic elapsed time, a regression test was added, and the full
  run was repeated. Video generation now rejects non-monotonic telemetry.
- X11 video PTS also drifted relative to ROS's monotonic elapsed time. The final capture records
  each real frame's monotonic timestamp and synchronizes by that index rather than assuming
  the raw MP4 frame rate is a reliable time source. No numerical robot animation substitutes for
  the screen capture.

## Interpretation and limits

Prompt phrasing affects predictions substantially. Short visual descriptions were chosen because they
avoid mixing visual state with the action to take. The expected labels were never provided to the model.
The 0.60 threshold was fixed before this experiment; it was not selected to fit a single confidence value.
These synthetic images were used in development. They are not an independent test set. A high route
accuracy on them would not establish real-world quality-control performance.

The initial run and raw screen recording remain under `output/runs/` for local audit. The submission ZIP
contains the final selected run and the prompt comparison, but omits host logs with local usernames.
