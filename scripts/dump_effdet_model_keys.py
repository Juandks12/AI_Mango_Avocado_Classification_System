from scripts.infer_effdet import build_effdet
m=build_effdet(3,'tf_efficientdet_d0')
for k in sorted(m.state_dict().keys()):
    if any(tok in k for tok in ('class_net','box_net','predict','class_predict','box_predict','class_head','box_head')):
        print(k)
print('total', len(m.state_dict()))
