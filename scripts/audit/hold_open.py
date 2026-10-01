"""Stand-in 'scorer' for serve_and_score.sh when the successor is served for an auditor, not scored.

serve_and_score.sh already does the three things an audit needs before any query -- repackage a text-only
Qwen3.5 save, serve it, wait out the GDN JIT -- and tears the server down when its scorer exits. This
scorer just writes a ready marker and blocks until the driver touches /tmp/audit/done (or 4 h pass), so
the server lives exactly as long as the audit."""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--out", required=True)
a, _ = ap.parse_known_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"ready": True, "base_url": a.base_url}, open(a.out, "w"))
t0 = time.time()
while not os.path.exists("/tmp/audit/done") and time.time() - t0 < 4 * 3600:
    time.sleep(5)
