"""Write a synthetic interval session as a FIT file (simulated data, not a recording): 8 reps of 45 s with 60 s
unlapped recoveries, 1 Hz records. Speed rises exponentially to a peak, then
holds; peak and hold speed fall slightly across reps (a built-in fatigue trend).

Usage: pip install fit-tool numpy
       python sample-data/make_synthetic_fit.py sample-data/synthetic-intervals.fit
"""
import sys, datetime as dt
import numpy as np
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.profile_type import FileType, Manufacturer

rng = np.random.default_rng(42)
out = sys.argv[1]
t0 = int(dt.datetime(2026, 9, 1, 9, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
REPS, REP_S, REC_S = 8, 45, 60

b = FitFileBuilder(auto_define=True, min_string_size=50)
fid = FileIdMessage()
fid.type = FileType.ACTIVITY
fid.manufacturer = Manufacturer.DEVELOPMENT.value
fid.product = 0
fid.time_created = t0
fid.serial_number = 0x1234
b.add(fid)

records, laps, t, dist = [], [], 0, 0.0
def rec(sec, v):
    global dist
    dist += v
    r = RecordMessage(); r.timestamp = t0 + sec * 1000; r.enhanced_speed = float(v); r.distance = dist
    return r
for k in range(REPS):
    peak = 6.4 - 0.05 * k            # m/s, small decline per rep
    tau = 2.2
    start = t
    speeds = []
    for s in range(REP_S + 1):
        v = 2.0 + (peak - 2.0) * (1 - np.exp(-s / tau))
        if s > 15:                   # later part of the rep fades more in later reps
            v -= 0.004 * k * (s - 15)
        v += rng.normal(0, 0.06)
        speeds.append(max(v, 0.5))
        records.append(rec(t, speeds[-1])); t += 1
    lap = LapMessage()
    lap.start_time = t0 + start * 1000
    lap.timestamp = t0 + (t - 1) * 1000
    lap.total_distance = float(sum(speeds))
    lap.enhanced_avg_speed = float(np.mean(speeds))
    laps.append(lap)
    for s in range(REC_S):           # easy jog recovery, not lapped
        records.append(rec(t, 1.6 + rng.normal(0, 0.05))); t += 1
b.add_all(records)
b.add_all(laps)
b.build().to_file(out)
print(f"wrote {out}: {REPS} reps, {len(records)} records")
