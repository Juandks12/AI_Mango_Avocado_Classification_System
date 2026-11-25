from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.infer_effdet import infer_effdet

if __name__ == '__main__':
    # Adjust these paths if needed
    model_path = Path('models/effdet_aguacate.pth')
    # pick a recent upload image
    img = Path('static/uploads')
    imgs = sorted(img.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not imgs:
        print('No images found in static/uploads')
    else:
        img_path = imgs[0]
        out_img = Path('static/outputs/test_effdet_out.jpg')
        out_json = Path('static/outputs/test_effdet_out.json')
        print('Running infer_effdet with', model_path, img_path)
        meta = infer_effdet(str(model_path), str(img_path), str(out_img), str(out_json), conf=0.03, effdet_name='tf_efficientdet_d0', class_names=['background','aguacate_optimo','aguacate_defecto'])
        print('Done. Meta:', meta)
