import json
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = PROJECT_ROOT / 'data' / 'aitod' / 'annotations' / 'aitod_test_v1_new.json'
DEFAULT_PRED = PROJECT_ROOT / 'work_dirs' / 'predictions.json'
DEFAULT_OUTPUT = PROJECT_ROOT / 'work_dirs' / 'predictions_filtered.json'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default=str(DEFAULT_GT),
                        help='Ground-truth annotation JSON (COCO format)')
    parser.add_argument('--pred', type=str, default=str(DEFAULT_PRED),
                        help='Prediction JSON to merge ids into')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output file for merged predictions')
    return parser.parse_args()


args = parse_args()
resFile = Path(args.gt)
with open(resFile, "r") as f:
    anns = json.loads(f.read())
anns = anns['images']

def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result

merge={}
for ann in anns:
    a={ann['file_name'][:-4]:ann['id']}
    merge.update(a)

# Driver code
# dict_1 = {'John': 15, 'Rick': 10, 'Misa' : 12 }
# dict_2 = {'Bonnie': 18,'Rick': 20,'Matt' : 16 }
# dict_3 = Merge(dict_1, dict_2)
# print(dict_3)

# annsImgIds = [ann['file_name'][:-4] for ann in anns]
# a = set(annsImgIds)
# {'area': 176, 'bbox': [618, 484, 22, 8], 'category_id': 5, 'id': 21218, 'image_id': 882, 'iscrowd': 0, 'segmentation': []}
# [{"file_name": "22766.png", "id": 0, "width": 800, "height": 800}, {"file_name": "2053__2298_1200.png", "id": 1, "width": 800, "height": 800}
resFile_2 = Path(args.pred)
with open(resFile_2, "r") as f2:
    anns_2 = json.loads(f2.read())
for ann2 in anns_2:
    ann2['image_id'] = merge.get(ann2['image_id'],0)
    # if merge.get(ann2['image_id'])==None:
    #      print(ann2['image_id'])
    #      exit()
# print(anns_2[0])
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(anns_2, f)
#  {'image_id': '9999984_00000_d_0000160__0_0', 'category_id': 5, 'bbox': [333.48, 494.464, 69.869, 34.831], 'score': 0.01973}
exit()
annsImgIds_2 = [ann2['image_id'] for ann2 in anns_2]
b = set(annsImgIds_2)
print(len(a))
print(len(b))
print(len(a&b))
print(len(a-b))
