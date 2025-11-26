import json
import os, sys, random, copy



def add_field_json(in_json, out_json, field = 'gender', optns = [0,1,2], optns_names = ["man",'woman', "child"],max_items = 100):
    with open(in_json) as json_data:
        d = json.load(json_data)
        json_data.close()

    images = d['images']
    annotations = d['annotations']
    new_annotations = []
    new_images = []
    count = 0
    all_image_id = []
    for a in annotations:
        ran = random.randint(0,len(optns)-1)
        a[field]= optns[ran]
        a[field+'_label'] = optns_names[ran]
        all_image_id.append(a['image_id'])
        new_annotations.append(copy.deepcopy(a))
        count = count + 1
        if(count > max_items):
            break

    count_im = 0
    for im in images:
        if(im['id'] in all_image_id):
            new_images.append(copy.deepcopy(im))

    d['annotations'] = new_annotations
    d['images'] = new_images
    d['user_categories']=[{'id':1,'labels': optns_names}]

    with open(out_json, 'w') as f:
        json.dump(d, f)




    print('OK')












if __name__ == "__main__":
    injson = "/home/borisef/data/coco/annotations/person_keypoints_val2017.json"
    outjson= "/home/borisef/data/coco/annotations/person_keypoints_with_gender.json"

    add_field_json(in_json=injson, out_json=outjson, field = 'gender', optns = [0,1,2],optns_names = ["man",'woman', "child"], max_items = 30)
    add_field_json(in_json=outjson, out_json=outjson, field='shape', optns=[0, 1], optns_names=["delta", 'non_delta'], max_items=30)