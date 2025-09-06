# parse_coco_pose.py
import os, json, csv, numpy as np, sys

ROOT = "./LINEMOD/cat"  # EDIT if needed
POSE_OUT = os.path.join(ROOT, "pose")
os.makedirs(POSE_OUT, exist_ok=True)

json_path = os.path.join(ROOT, "test.json")
if not os.path.exists(json_path):
    raise SystemExit(f"No test.json at {json_path}")

data = json.load(open(json_path, 'r'))

# Build image id -> filename mapping
images = data.get('images', [])
imgid2name = {}
for img in images:
    fname = img.get('file_name') or img.get('file') or img.get('filename')
    if fname is None:
        continue
    # take basename only (handles "data/linemod/cat/JPEGImages/000000.jpg")
    b = os.path.basename(fname)
    imgid2name[img.get('id')] = b

# Build annotations list grouped by image_id
anns = data.get('annotations', [])
by_img = {}
for a in anns:
    img_id = a.get('image_id') or a.get('image') or a.get('img_id')
    if img_id is None:
        continue
    by_img.setdefault(img_id, []).append(a)

def nums_from_obj(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, np.ndarray)):
        flat = []
        for x in obj:
            if isinstance(x, (list, tuple, np.ndarray)):
                flat.extend([float(v) for v in np.array(x).reshape(-1).tolist()])
            else:
                try:
                    flat.append(float(x))
                except:
                    pass
        return flat
    if isinstance(obj, (int,float)):
        return [float(obj)]
    if isinstance(obj, str):
        parts = obj.replace(',', ' ').split()
        res = []
        for p in parts:
            try:
                res.append(float(p))
            except:
                pass
        return res
    return []

def try_extract_pose_from_ann(ann):
    # common keys to try (ordered)
    # returns (Rflat(9), tflat(3)) or None
    candidates = []
    # direct fields that might contain R/t or flattened arrays
    for k in ('R','r','rotation','rot','R_mat','rotation_matrix','cam_R_m2c','cam_R','Rt','pose','pose_rt','cam_Rt','pose_mat'):
        if k in ann:
            candidates.append((k, ann[k]))
    for k in ('t','translation','trans','cam_t_m2c','cam_t','T'):
        if k in ann:
            candidates.append((k, ann[k]))
    # also try a combined 'pose' or 'rt' or 'transformation' field
    for k in ('pose','pose_rt','rt','transformation','transform'):
        if k in ann:
            candidates.append((k, ann[k]))
    # brute-force: check all numeric-valued fields
    for k,v in ann.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(nums_from_obj(v)) >= 12:
            nums = nums_from_obj(v)
            return nums[:9], nums[9:12]

    # Try direct separate fields: R (matrix-like) and t
    # If ann contains 'R' and 't' separately (or cam_R_m2c / cam_t_m2c)
    R_keys = ['R','r','rotation','cam_R_m2c','cam_R','R_mat','rotation_matrix']
    t_keys = ['t','translation','cam_t_m2c','cam_t']
    Rflat = None; tflat = None
    for rk in R_keys:
        if rk in ann:
            Rflat = nums_from_obj(ann[rk])
            break
    for tk in t_keys:
        if tk in ann:
            tflat = nums_from_obj(ann[tk])
            break
    if Rflat is not None and tflat is not None and len(Rflat) >= 9 and len(tflat) >= 3:
        return Rflat[:9], tflat[:3]

    # Some annotations may use 'rotation' as 3 Rodrigues numbers + t (3) => 6 numbers
    # or quaternion + t (4+3). We don't auto-convert quaternion here.
    # Try detect rodigues + t as 6-length list
    for k in ('rotation','rot'):
        if k in ann:
            rotnums = nums_from_obj(ann[k])
            if len(rotnums) >= 3 and ('t' in ann or 'translation' in ann or 'trans' in ann):
                tnums = nums_from_obj(ann.get('t') or ann.get('translation') or ann.get('trans'))
                if len(tnums) >= 3:
                    # convert Rodrigues to R
                    try:
                        import cv2
                        rvec = np.array(rotnums[:3], dtype=np.float32).reshape(3,1)
                        Rmat, _ = cv2.Rodrigues(rvec)
                        return Rmat.reshape(-1).tolist(), tnums[:3]
                    except Exception:
                        pass

    return None

rows = []
skipped = []
import cv2  # need for Rodrigues conversion if used

for img_id, fname in imgid2name.items():
    anns_for_img = by_img.get(img_id, [])
    # If multiple annotations, pick first that yields pose
    found = False
    for ann in anns_for_img:
        res = try_extract_pose_from_ann(ann)
        if res is not None:
            Rf, tf = res
            # write pose file
            outname = os.path.join(POSE_OUT, os.path.splitext(fname)[0] + ".txt")
            with open(outname, 'w') as fo:
                fo.write(" ".join([str(float(x)) for x in (Rf[:9] + tf[:3])]))
            rows.append((fname, fname, " ".join([str(float(x)) for x in Rf[:9]]), " ".join([str(float(x)) for x in tf[:3]])))
            found = True
            break
    if not found:
        skipped.append(fname)

# Write CSV
csvf = os.path.join(ROOT, "train_split.csv")
with open(csvf, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["image","mask","R","t"])
    for r in rows:
        w.writerow(r)

print(f"Wrote {csvf} with {len(rows)} rows")
if skipped:
    print("Skipped images (no parseable pose):", len(skipped))
    print("Example skipped:", skipped[:10])
    # show sample annotation keys for first skipped image for debugging
    example = skipped[0]
    # find its id
    ex_id = None
    for k,v in imgid2name.items():
        if v == example:
            ex_id = k; break
    if ex_id is not None:
        ex_anns = by_img.get(ex_id, [])
        if ex_anns:
            print("First skipped annotation keys:", list(ex_anns[0].keys()))
        else:
            print("No annotations entries for that image.")
