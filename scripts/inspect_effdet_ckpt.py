import torch
p='models/effdet_aguacate.pth'
raw=torch.load(p,map_location='cpu')
# attempt to extract state_dict
if isinstance(raw, dict) and 'state_dict' in raw:
    sd=raw['state_dict']
elif isinstance(raw, dict) and 'model' in raw and isinstance(raw['model'], dict):
    sd=raw['model']
else:
    sd=raw

# strip prefixes
out={}
for k,v in sd.items():
    new=k
    for pfx in ('module.model.','model.','module.'):
        if new.startswith(pfx):
            new=new[len(pfx):]
            break
    out[new]=v

for k in sorted(out.keys()):
    if any(tok in k for tok in ('class_net','box_net','predict','class_predict','anchors','class_head','box_head','score')):
        try:
            print(k, getattr(out[k],'shape', None))
        except Exception:
            print(k, type(out[k]))

print('Total keys:', len(out))
