import json
p='D:/TODO2/mango.v1i.coco/train/_annotations.coco.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
print('images:', len(data.get('images',[])))
print('annotations:', len(data.get('annotations',[])))
for i in range(5):
    ann=data['annotations'][i]
    print('ann bbox sample:', ann.get('bbox'))
    img_id=ann.get('image_id')
    img=[im for im in data['images'] if im['id']==img_id][0]
    print('img width,height:', img.get('width'), img.get('height'))
    break
