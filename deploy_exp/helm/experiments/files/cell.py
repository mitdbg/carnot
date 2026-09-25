"""Tiny helper the pod scripts call to read THIS pod's cell out of /manifest/cells.json.

    cell.py <field>              print a string field (lists/dicts as JSON)
    cell.py --argv               print argv NUL-separated (for `readarray -d ''`)
    cell.py --data               print one "src<TAB>dest<TAB>include<TAB>exclude" line per data entry
                                 (dest defaults to src; include/exclude are optional aws s3 sync globs)

The index comes from JOB_COMPLETION_INDEX (set by the Indexed Job); CELLS_JSON overrides the manifest path.
Plain stdlib so it runs with the image's python before anything else is set up."""
import json
import os
import sys

cells = json.load(open(os.environ.get("CELLS_JSON", "/manifest/cells.json")))
idx = int(os.environ["JOB_COMPLETION_INDEX"])
if not 0 <= idx < len(cells):
    sys.exit(f"cell.py: JOB_COMPLETION_INDEX={idx} out of range for {len(cells)} cells")
cell = cells[idx]

what = sys.argv[1]
if what == "--argv":
    sys.stdout.write("\0".join(str(a) for a in cell["argv"]))
elif what == "--data":
    for entry in cell.get("data", []):
        if isinstance(entry, str):
            entry = {"src": entry}
        print("\t".join([entry["src"], entry.get("dest", entry["src"]), entry.get("include", ""), entry.get("exclude", "")]))
else:
    v = cell.get(what, "")
    print(v if isinstance(v, str) else json.dumps(v))
