import torch
from effdet import create_model
import traceback

ckpt_path = 'runs/aguacate.v1i.coco/effdet/best.pth'
state = torch.load(ckpt_path, map_location='cpu')
print('checkpoint type:', type(state))
if isinstance(state, dict):
    keys_sample = list(state.keys())[:20]
    print('sample keys:', keys_sample)

variants = [
    'tf_efficientdet_d0','tf_efficientdet_d1','tf_efficientdet_d2','tf_efficientdet_d3',
    'tf_efficientdet_d4','tf_efficientdet_d5','tf_efficientdet_d6'
]

for name in variants:
    print('\nTrying variant:', name)
    try:
        m = create_model(name, bench_task='predict', pretrained=False)
    except Exception as e:
        print('  create_model failed:', e)
        continue
    conf = getattr(m,'config',None)
    img_size = None
    if conf is not None:
        img_size = getattr(conf,'image_size',None)
    if img_size and isinstance(img_size,(list,tuple)):
        h,w = int(img_size[1]), int(img_size[0])
    else:
        h,w = 640,640
    # try loading state: try different key mappings
    st = state
    if isinstance(st, dict):
        keys = list(st.keys())
        if keys and all(k.startswith('model.') for k in keys[:min(20,len(keys)):]):
            st = {k.replace('model.','',1):v for k,v in st.items()}
    try:
        m.load_state_dict(st)
        print('  load_state_dict ok (strict=True)')
    except Exception as e:
        try:
            m.load_state_dict(st, strict=False)
            print('  load_state_dict ok (strict=False)')
        except Exception as e2:
            print('  load failed:', e2)
            continue
    m.eval()
    x = torch.randn(1,3,h,w)
    try:
        out = m(x)
        print('  forward OK, output type:', type(out))
    except Exception as e:
        print('  forward failed:', e)
        traceback.print_exc()

print('\nDone')
