"""CSV I/O, EpisodeRow dataclass, and summary table for sim-to-sim evaluation."""

import csv
from dataclasses import dataclass, fields, astuple
from pathlib import Path


@dataclass
class EpisodeRow:
    sim: str
    seed: int
    survived: bool
    tracking_error_mean: float   # mean |linvel_x - 1.0| over alive steps (m/s)
    base_height_dev_mean: float  # mean |base_z - 0.50| over alive steps (m)
    feet_force_rms: float        # sqrt(mean ||f_foot||^2) over alive steps × 4 feet (N)
    episode_seconds: float       # how long the episode lasted (may be < 30 s if fallen)
    fall_timestep: int           # -1 if survived; else 0-based step index when termination triggered


_FIELDNAMES = [f.name for f in fields(EpisodeRow)]


def read_existing(csv_path: Path) -> set[tuple[str, int]]:
    """Return set of (sim, seed) pairs already present in the CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {(row["sim"], int(row["seed"])) for row in reader}


def append_rows(csv_path: Path, rows: list[EpisodeRow]) -> None:
    """Append rows to the CSV, writing the header only when the file is new."""
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_FIELDNAMES)
        for row in rows:
            writer.writerow(astuple(row))


def print_summary(csv_path: Path) -> None:
    """Group rows by sim, print mean ± std for each metric, check done criteria."""
    if not csv_path.exists():
        print("No results file found.")
        return

    rows: list[EpisodeRow] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(EpisodeRow(
                sim=r["sim"],
                seed=int(r["seed"]),
                survived=r["survived"].lower() == "true",
                tracking_error_mean=float(r["tracking_error_mean"]),
                base_height_dev_mean=float(r["base_height_dev_mean"]),
                feet_force_rms=float(r["feet_force_rms"]),
                episode_seconds=float(r["episode_seconds"]),
                fall_timestep=int(r.get("fall_timestep", -1)),
            ))

    # Group by sim.
    by_sim: dict[str, list[EpisodeRow]] = {}
    for r in rows:
        by_sim.setdefault(r.sim, []).append(r)

    import statistics

    sep = "─" * 80
    print(f"\n{'SIM-TO-SIM EVALUATION SUMMARY':^80}")
    print(sep)
    fmt = "{:<18} {:>8} {:>16} {:>16} {:>14}"
    print(fmt.format("Sim", "Surv%", "TrackErr(m/s)", "HeightDev(m)", "FeetForce(N)"))
    print(sep)

    per_sim_survival: dict[str, float] = {}
    per_sim_tracking: dict[str, float] = {}

    for sim_name, sim_rows in sorted(by_sim.items()):
        n = len(sim_rows)
        surv_pct = 100.0 * sum(r.survived for r in sim_rows) / n
        track_vals = [r.tracking_error_mean for r in sim_rows]
        height_vals = [r.base_height_dev_mean for r in sim_rows]
        force_vals = [r.feet_force_rms for r in sim_rows]

        def _fmt(vals):
            if len(vals) < 2:
                return f"{vals[0]:.3f}"
            return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"

        print(fmt.format(
            f"{sim_name} (n={n})",
            f"{surv_pct:.1f}%",
            _fmt(track_vals),
            _fmt(height_vals),
            _fmt(force_vals),
        ))
        per_sim_survival[sim_name] = surv_pct
        per_sim_tracking[sim_name] = statistics.mean(track_vals)

    print(sep)

    # M5 done criteria.
    if len(per_sim_survival) >= 2:
        surv_drop = max(per_sim_survival.values()) - min(per_sim_survival.values())
        track_drift = max(per_sim_tracking.values()) - min(per_sim_tracking.values())

        c1_pass = surv_drop < 15.0
        c2_pass = track_drift < 0.2

        def _verdict(ok):
            return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"

        print(f"  M5-C1  Survival drop across sims < 15 pp :  {surv_drop:.1f} pp   {_verdict(c1_pass)}")
        print(f"  M5-C2  Tracking drift across sims < 0.20 :  {track_drift:.3f} m/s  {_verdict(c2_pass)}")
        print(sep)

    # Fall-timestep analysis (distinguishes "bad initial contact" from "accumulated drift").
    fall_rows_any = [r for r in rows if not r.survived and r.fall_timestep >= 0]
    if fall_rows_any:
        print(f"\n{'FALL TIMING ANALYSIS':^80}")
        print(sep)
        _buckets = [(0, 50), (51, 200), (201, 500), (501, 10**9)]
        _bucket_labels = ["[0–50]", "[51–200]", "[201–500]", "[500+]"]
        hdr = "{:<18} {:>6} " + " ".join(f"{lb:>12}" for lb in _bucket_labels) + "  {:>10}  {:>14}"
        print(hdr.format("Sim", "Falls", *_bucket_labels, "MeanStep", "ImmediateFalls%"))
        print(sep)

        for sim_name, sim_rows in sorted(by_sim.items()):
            fallen = [r for r in sim_rows if not r.survived and r.fall_timestep >= 0]
            if not fallen:
                counts = [0] * len(_buckets)
                print("{:<18} {:>6} ".format(f"{sim_name}", 0) +
                      " ".join(f"{'0':>12}" for _ in _buckets) +
                      f"  {'—':>10}  {'—':>14}")
                continue

            counts = []
            for lo, hi in _buckets:
                counts.append(sum(1 for r in fallen if lo <= r.fall_timestep <= hi))
            mean_step = statistics.mean(r.fall_timestep for r in fallen)
            imm_pct = 100.0 * counts[0] / len(fallen)

            row_str = "{:<18} {:>6} ".format(f"{sim_name}", len(fallen))
            row_str += " ".join(f"{c:>12}" for c in counts)
            row_str += f"  {mean_step:>10.1f}  {imm_pct:>13.1f}%"
            print(row_str)

        print(sep)
