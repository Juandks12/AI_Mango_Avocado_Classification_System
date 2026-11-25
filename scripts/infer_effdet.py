# scripts/infer_effdet.py
import torch
from pathlib import Path
from datetime import datetime
import json
from PIL import Image, ImageDraw, ImageFont

import torchvision

try:
    from effdet import create_model as effdet_create_model
    HAS = True
except Exception:
    HAS = False

def build_effdet(num_classes=3, name="tf_efficientdet_d0"):
    if not HAS:
        raise RuntimeError("effdet not installed")
    return effdet_create_model(name, bench_task='predict', num_classes=num_classes, pretrained=False)

def infer_effdet(model_path, img_path, out_img_path, out_json_path, conf=0.85, effdet_name="tf_efficientdet_d0", class_names=None):
    if not HAS:
        raise RuntimeError("effdet/timm not installed in environment")
    # Use CPU for EffDet by default to avoid runtime mismatches with GPU/timm versions.
    # Manual tests showed the CPU path succeeds for this checkpoint exported by Roboflow.
    device = torch.device("cpu")

    # Load checkpoint (may be a plain state_dict OrderedDict)
    raw_state = torch.load(model_path, map_location="cpu")

    # If the checkpoint is a mapping with keys like 'model.*', extract that mapping
    if isinstance(raw_state, dict) and not any(k.startswith('module.') for k in raw_state.keys()):
        # raw_state may itself be a state_dict (OrderedDict) where keys include 'model.' prefix
        state_dict = raw_state
    else:
        # fallback: try common keys
        if isinstance(raw_state, dict) and 'state_dict' in raw_state:
            state_dict = raw_state['state_dict']
        elif isinstance(raw_state, dict) and 'model' in raw_state and isinstance(raw_state['model'], dict):
            state_dict = raw_state['model']
        else:
            state_dict = raw_state

    # If keys are like 'model.<rest>', strip the 'model.' prefix
    stripped = {}
    for k, v in list(state_dict.items()):
        if k.startswith('model.'):
            stripped[k[len('model.'):]] = v
        elif k.startswith('module.model.'):
            stripped[k[len('module.model.'):]] = v
        elif k.startswith('module.'):
            stripped[k[len('module.'):]] = v
        else:
            stripped[k] = v

    state_dict = stripped

    # Try to infer number of classes from class head bias if present
    num_classes_from_ckpt = None
    for k in state_dict.keys():
        if k.endswith('class_net.predict.conv_pw.bias') or k.endswith('class_net.predict.conv_pw.weight'):
            # bias shape gives out_channels = num_anchors * num_classes
            v = state_dict[k]
            try:
                out_ch = v.shape[0] if hasattr(v, 'shape') and len(v.shape) > 0 else None
            except Exception:
                out_ch = None
            if out_ch is not None and out_ch > 0:
                # common number of anchors per location is 9 for effdet; try to divide
                if out_ch % 9 == 0:
                    num_classes_from_ckpt = out_ch // 9
                else:
                    # try common divisors
                    for cand in [3,4,5,2]:
                        if out_ch % cand == 0 and out_ch // cand <= 100:
                            num_classes_from_ckpt = out_ch // cand
                            break
            break

    target_num_classes = num_classes_from_ckpt or 3

    # Determine anchor count in checkpoint if available to match variant
    ckpt_anchor_count = None
    if 'anchors.boxes' in state_dict:
        try:
            ckpt_anchor_count = int(state_dict['anchors.boxes'].shape[0])
        except Exception:
            ckpt_anchor_count = None

    # Candidate model names to try if the requested name fails at forward time
    candidates = [effdet_name] if effdet_name else []
    candidates += [
        'tf_efficientdet_d0', 'tf_efficientdet_d1', 'tf_efficientdet_d2',
        'tf_efficientdet_d3', 'tf_efficientdet_d4', 'tf_efficientdet_d5'
    ]

    def match_and_load(m, ckpt_sd):
        """Match checkpoint tensors to model state_dict keys by shape and suffix similarity.
        Returns (matched_count, unmatched_model_keys, unmatched_ckpt_keys)
        and leaves the matched tensors loaded into the model with strict=False.
        """
        model_sd = m.state_dict()
        # prepare stripped forms of ckpt keys
        def strip_prefixes(k):
            for p in ('module.model.', 'model.', 'module.'):
                if k.startswith(p):
                    return k[len(p):]
            return k

        ckpt_variants = {}
        for k, v in ckpt_sd.items():
            base = strip_prefixes(k)
            ckpt_variants.setdefault(base, []).append((k, v))

        load_sd = {}
        matched = []
        unmatched_model_keys = []
        used_ckpt_keys = set()

        for mkey, mval in model_sd.items():
            mshape = getattr(mval, 'shape', None)
            # candidate ckpt bases to consider: exact base name or suffix matches
            best_candidate = None
            best_score = -1
            # iterate ckpt variants where shapes match
            for base, entries in ckpt_variants.items():
                for ckpt_full, ckpt_val in entries:
                    if getattr(ckpt_val, 'shape', None) != mshape:
                        continue
                    # compute suffix-match score by comparing dotted components from the end
                    bs = base.split('.')
                    ms = mkey.split('.')
                    score = 0
                    for a, b in zip(reversed(bs), reversed(ms)):
                        if a == b:
                            score += 1
                        else:
                            break
                    if score > best_score:
                        best_score = score
                        best_candidate = (ckpt_full, ckpt_val)
            if best_candidate is not None:
                load_sd[mkey] = best_candidate[1]
                matched.append((mkey, best_candidate[0]))
                used_ckpt_keys.add(best_candidate[0])
            else:
                unmatched_model_keys.append(mkey)

        try:
            m.load_state_dict(load_sd, strict=False)
        except Exception:
            # final attempt: load loose whole ckpt
            try:
                m.load_state_dict(ckpt_sd, strict=False)
            except Exception:
                pass

        unmatched_ckpt_keys = [k for k in ckpt_sd.keys() if k not in used_ckpt_keys]
        return len(matched), unmatched_model_keys, unmatched_ckpt_keys, matched

    def try_build_and_load(name):
        try:
            m = build_effdet(target_num_classes, name)
        except Exception:
            return None, 'build_failed'
        m.to(device)
        m.eval()
        matched_count, um_model, um_ckpt, matched_pairs = match_and_load(m, state_dict)
        # perform a lightweight forward test using the model's configured image size
        ff_err = None
        try:
            cfg = getattr(m, 'config', None)
            img_size = None
            if cfg is not None:
                img_size = getattr(cfg, 'image_size', None)
                if img_size is None and hasattr(cfg, 'get'):
                    img_size = cfg.get('image_size')
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                h, w = int(img_size[0]), int(img_size[1])
            elif isinstance(img_size, int):
                h = w = int(img_size)
            else:
                h = w = 640

            dummy = torch.zeros(1, 3, h, w, device=device)
            with torch.no_grad():
                _ = m(dummy)
        except Exception as e:
            ff_err = e

        status = {'status': 'ok', 'matched_count': matched_count, 'unmatched_model': um_model, 'unmatched_ckpt': um_ckpt, 'pairs': matched_pairs}
        if ff_err:
            # attach forward error repr for diagnostics
            try:
                status['forward_error'] = repr(ff_err)
            except Exception:
                status['forward_error'] = str(type(ff_err))
            return None, status
        return m, status

    model = None
    last_err = None
    selected_status = None
    # If checkpoint has anchors info, try to preselect the variant by matching anchor counts
    best_candidate = None
    if ckpt_anchor_count is not None:
        for cand in candidates:
            try:
                mcand = build_effdet(target_num_classes, cand)
                if hasattr(mcand, 'anchors') and hasattr(mcand.anchors, 'boxes'):
                    try:
                        acount = int(getattr(mcand.anchors, 'boxes').shape[0])
                        if acount == ckpt_anchor_count:
                            best_candidate = cand
                            break
                    except Exception:
                        pass
            except Exception:
                continue

    if best_candidate:
        m, status = try_build_and_load(best_candidate)
        selected_status = status
        if m is not None:
            model = m

    if model is None:
        for cand in candidates:
            m, status = try_build_and_load(cand)
            selected_status = status
            if m is None:
                last_err = status
                continue
            model = m
            break

    if model is None:
        # fallback: build requested name with default classes and try to load loosely
        model = build_effdet(target_num_classes, effdet_name)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # try copying by matching shapes
            model_sd = model.state_dict()
            load_sd = {}
            for k_ckpt, v in state_dict.items():
                if k_ckpt in model_sd and getattr(v, 'shape', None) == getattr(model_sd[k_ckpt], 'shape', None):
                    load_sd[k_ckpt] = v
            model.load_state_dict(load_sd, strict=False)

    model.to(device).eval()

    # write selection/debug info
    try:
        sel_debug = Path(out_json_path).with_suffix('.effdet_selected.txt')
        with open(sel_debug, 'w', encoding='utf8') as sf:
            sf.write(f'ckpt_anchor_count={ckpt_anchor_count}\n')
            try:
                sf.write(f'selected_model_image_size={getattr(model.config, "image_size", None)}\n')
            except Exception:
                sf.write('selected_model_image_size=<err>\n')
            try:
                # include matched_count and pairs if available from loader
                if isinstance(selected_status, dict):
                    sf.write(f"matched_param_count={selected_status.get('matched_count')}\n")
                    try:
                        sf.write('matched_pairs:\n')
                        for mk, ck in selected_status.get('pairs', [])[:200]:
                            sf.write(f"  {mk} <- {ck}\n")
                    except Exception:
                        pass
                else:
                    # fallback: count how many params were matched and present in model
                    matched = 0
                    for k in state_dict.keys():
                        if k in model.state_dict() and getattr(state_dict[k], 'shape', None) == getattr(model.state_dict()[k], 'shape', None):
                            matched += 1
                    sf.write(f'matched_param_count={matched}\n')
            except Exception:
                pass
    except Exception:
        pass

    img = Image.open(img_path).convert("RGB")
    # Try to resize image to model's expected input size when available to avoid
    # mismatched feature map shapes during forward.
    expected_size = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            # effdet config may expose attribute image_size or dict-like access
            expected_size = getattr(cfg, "image_size", None)
            if expected_size is None and hasattr(cfg, "get"):
                expected_size = cfg.get("image_size")
        except Exception:
            expected_size = None
    if expected_size is None:
        expected_size = getattr(model, "image_size", None)
    # normalize to (h,w)
    resize_to = None
    try:
        if isinstance(expected_size, int):
            resize_to = (expected_size, expected_size)
        elif isinstance(expected_size, (list, tuple)) and len(expected_size) == 2:
            resize_to = tuple(expected_size)
    except Exception:
        resize_to = None
    # Force exact resize to expected config size when available. Use conservative defaults.
    target_h = target_w = None
    try:
        if resize_to is not None:
            target_h = int(resize_to[0]); target_w = int(resize_to[1])
        else:
            target_h = target_w = 640
    except Exception:
        target_h = target_w = 640

    try:
        # PIL expects (width, height)
        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
    except Exception:
        try:
            img = torchvision.transforms.functional.resize(img, (target_h, target_w))
        except Exception:
            pass

    t = torchvision.transforms.functional.to_tensor(img).to(device)
    # ensure tensor matches expected model input size; if not, interpolate to exact size
    if resize_to is not None:
        try:
            target_h = int(resize_to[0]); target_w = int(resize_to[1])
            if t.shape[1] != target_h or t.shape[2] != target_w:
                import torch.nn.functional as F
                t = F.interpolate(t.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
        except Exception:
            pass
    # pass a list of images (like torchvision detection models) to allow variable sizes
    with torch.no_grad():
        # First try: pass a batched tensor (most effdet versions expect a tensor batch)
        out = None
        try:
            res = model(t.unsqueeze(0))
            try:
                out = res[0]
            except Exception:
                out = res
        except Exception as e_tensor:
            # record the tensor-call failure
            debug_path = Path(out_json_path).with_suffix('.effdet_debug_tensor.txt')
            try:
                with open(debug_path, 'w', encoding='utf8') as df:
                    import traceback
                    df.write('TENSOR CALL FAILED:\n')
                    traceback.print_exc()
                    df.write('\nException repr:\n' + repr(e_tensor))
            except Exception:
                pass
            # Attempt to collect backbone feature shapes and model metadata for diagnosis
            try:
                core = getattr(model, 'model', model)
                info_path = Path(out_json_path).with_suffix('.effdet_feature_shapes.txt')
                with open(info_path, 'w', encoding='utf8') as fh:
                    fh.write('INPUT TENSOR SHAPE: ' + repr(t.unsqueeze(0).shape) + '\n')
                    try:
                        cfg = getattr(core, 'config', None)
                        fh.write('MODEL CONFIG image_size: ' + repr(getattr(cfg, 'image_size', None)) + '\n')
                    except Exception:
                        pass
                    try:
                        anchors = getattr(core, 'anchors', None)
                        if anchors is not None and hasattr(anchors, 'boxes'):
                            fh.write('anchors.boxes.shape: ' + repr(getattr(anchors, 'boxes').shape) + '\n')
                    except Exception:
                        pass
                    try:
                        backbone = getattr(core, 'backbone', None)
                        if backbone is not None:
                            with torch.no_grad():
                                bf = backbone(t.unsqueeze(0))
                            fh.write('BACKBONE OUTPUT TYPE: ' + repr(type(bf)) + '\n')
                            if isinstance(bf, (list, tuple)):
                                for i, b in enumerate(bf):
                                    try:
                                        fh.write(f'  feat[{i}].shape: ' + repr(getattr(b, 'shape', None)) + '\n')
                                    except Exception:
                                        fh.write(f'  feat[{i}].repr_failed\n')
                            elif isinstance(bf, dict):
                                for kkk, vvv in bf.items():
                                    fh.write(f'  feat[{kkk}].shape: ' + repr(getattr(vvv, 'shape', None)) + '\n')
                            else:
                                fh.write('  BACKBONE OUTPUT UNHANDLED TYPE\n')
                    except Exception as _:
                        fh.write('BACKBONE SHAPE DUMP FAILED\n')
            except Exception:
                pass
            # Fallback: try calling with a list (some wrappers expect list)
            try:
                res = model([t])
                try:
                    out = res[0]
                except Exception:
                    out = res
            except Exception as e_list:
                # write debug file with exception info for list call too
                debug_path = Path(out_json_path).with_suffix('.effdet_debug_list.txt')
                try:
                    with open(debug_path, 'w', encoding='utf8') as df:
                        import traceback
                        df.write('LIST CALL FAILED:\n')
                        traceback.print_exc()
                        df.write('\nException repr:\n' + repr(e_list))
                except Exception:
                    pass
                # re-raise original tensor exception to surface error to user
                raise e_tensor

        # write a short debug dump of the returned structure to help troubleshooting
        try:
            debug_path = Path(out_json_path).with_suffix('.effdet_out_debug.txt')
            with open(debug_path, 'w', encoding='utf8') as df:
                df.write('TYPE: ' + repr(type(out)) + '\n')
                try:
                    df.write('REPR: ' + repr(out)[:10000] + '\n')
                except Exception:
                    df.write('REPR: <unrepresentable>\n')
        except Exception:
            pass

    # Try to parse output in different possible formats (effdet has variations)
    boxes = scores = labels = None

    # Some model wrappers return a single Tensor of shape (N,6) where
    # columns are [x1, y1, x2, y2, score, label]. Handle that case first.
    try:
        import torch as _torch
        if isinstance(out, _torch.Tensor):
            try:
                arr = out.detach().cpu()
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    boxes = arr[:, :4]
                    scores = arr[:, 4]
                    labels = arr[:, 5].long()
                elif arr.ndim == 1 and arr.numel() % 6 == 0:
                    arr = arr.view(-1, 6)
                    boxes = arr[:, :4]
                    scores = arr[:, 4]
                    labels = arr[:, 5].long()
            except Exception:
                boxes = scores = labels = None
    except Exception:
        pass

    def tensor_to_list(x):
        try:
            return x.detach().cpu().numpy().tolist()
        except Exception:
            # if it's already a list of numbers
            try:
                return [float(v) for v in x]
            except Exception:
                return []

    # out can be a dict-like, a list/tuple whose first element is dict-like,
    # or a simple tuple of (boxes, scores, labels)
    if isinstance(out, dict):
        det = out.get("detections") or out
        boxes = det.get("boxes") if isinstance(det, dict) else None
        scores = det.get("scores") if isinstance(det, dict) else None
        labels = det.get("labels") if isinstance(det, dict) else None
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict):
            det = first.get("detections") or first
            boxes = det.get("boxes") if isinstance(det, dict) else None
            scores = det.get("scores") if isinstance(det, dict) else None
            labels = det.get("labels") if isinstance(det, dict) else None
        elif hasattr(first, "boxes"):
            # object with attributes (like some model wrappers)
            try:
                boxes = first.boxes
                scores = first.scores if hasattr(first, "scores") else None
                labels = first.labels if hasattr(first, "labels") else None
            except Exception:
                boxes = scores = labels = None
        else:
            # maybe returned as tuple (boxes, scores, labels)
            if len(out) >= 3:
                boxes, scores, labels = out[0], out[1], out[2]

    # Convert tensors to lists safely
    try:
        if boxes is not None and hasattr(boxes, "detach"):
            boxes = tensor_to_list(boxes)
        elif isinstance(boxes, (list, tuple)) and len(boxes) > 0 and hasattr(boxes[0], "detach"):
            # list of tensors per image/detection -> convert first element
            boxes = tensor_to_list(boxes[0])
        elif not isinstance(boxes, list):
            boxes = []

        if scores is not None and hasattr(scores, "detach"):
            scores = tensor_to_list(scores)
        elif isinstance(scores, (list, tuple)) and len(scores) > 0 and hasattr(scores[0], "detach"):
            scores = tensor_to_list(scores[0])
        elif not isinstance(scores, list):
            scores = []

        if labels is not None and hasattr(labels, "detach"):
            labels = tensor_to_list(labels)
        elif isinstance(labels, (list, tuple)) and len(labels) > 0 and hasattr(labels[0], "detach"):
            labels = tensor_to_list(labels[0])
        elif not isinstance(labels, list):
            labels = []
    except Exception:
        boxes, scores, labels = [], [], []

    preds = []
    for i, (b,l,s) in enumerate(zip(boxes, labels, scores), start=1):
        if float(s) < conf: continue
        x1,y1,x2,y2 = [float(x) for x in b]
        # clip to image bounds (draw_img is resized image)
        try:
            img_w, img_h = img.size
            x1 = max(0.0, min(x1, float(img_w)))
            y1 = max(0.0, min(y1, float(img_h)))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))
        except Exception:
            pass
        w = x2-x1; h = y2-y1
        if w <= 1 or h <= 1:
            continue
        cname = None
        try:
            if class_names and 0 <= int(l) < len(class_names):
                cname = class_names[int(l)]
        except Exception:
            cname = None
        if not cname:
            cname = f"class_{int(l)}"
        preds.append({
            "id": i,
            "category_id": int(l),
            "category_name": cname,
            "bbox": [x1,y1,w,h],
            "score": float(s)
        })

    # draw
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    for p in preds:
        x,y,w,h = p["bbox"]
        draw.rectangle([x,y,x+w,y+h], outline="red", width=2)
        draw.text((x,y-12), f"{p['category_name']} {p['score']:.2f}", fill="white", font=font)

    Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
    draw_img.save(out_img_path)

    meta = {
        "input_path": str(img_path),
        "model_path": str(model_path),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "num_detections": len(preds),
        "predictions": preds
    }
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
