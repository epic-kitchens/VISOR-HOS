'''
Convert VISOR annotation to COCO format
'''

from ast import Pass
import json, glob, os,shutil, pdb, random, cv2, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from data_util import *
random.seed(0)

# COCO main structure -------------------------------------------------------------------------
coco = {
    # "info": {...},
    # "licenses": [...],
    # "images": [...],
    # "annotations": [...],
    # "categories": [...], 
    # "segment_info": [...]
}

epick_visor_info = {
    "description": "EPIC-KITCHENS VISOR",
    "url": "https://epic-kitchens.github.io/VISOR/",
    "version": "1.0",
    "year": 2022,
    "contributor": "Ahmad Darkhalil*, Dandan Shan*, Bin Zhu*, Jian Ma*, Amlan Kar, Richard E.L. Higgins, Sanja Fidler, David F. Fouhey, Dima Damen"
}

licenses = [
    # TODO: add license?
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }
]


images = [
    # {
    #    "id": 397133,
    #     "file_name": "000000397133.jpg",
    #     "height": 427,
    #     "width": 640
    # }, ...
]


color_ls = [(0, 90, 181), (220, 50, 32)] 
glove_ls = ['oven glove', 'gloves', 'rubber glove', 'left glove', 'right glove', 'glove']

if __name__ == '__main__':
    
    # python gen_coco_format.py --epick_visor_store=/nfs/turbo/fouheyUnrep/dandans/epick_visor_camera_ready/epick_visor/GroundTruth-SparseAnnotations --mode=active --split train val test
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epick_visor_store', type=str, required=True, default='/path/to/epick_visor/GroundTruth-SparseAnnotations', help='Folder saving EPIC-KITCHENS VISOR data.')
    # parser.add_argument('--save_folder', type=str, help='Folder name for the version of annotation.')
    parser.add_argument('--vis', action='store_true', help='Generate visualization or not.')
    parser.add_argument('--num', type=int, default=None, help='Number of jsons to process.')
    parser.add_argument('--copy_img', action='store_true', help='Whether to copy image.')
    parser.add_argument('--unzip_img', action='store_true', help='Whether to unzip image.')
    parser.add_argument('--split', nargs='+', required=True, help='Which split to generate COCO annotations.')
    parser.add_argument('--mode', type=str, required=True, help='[hos, active, all, handside, contact]')
    parser.add_argument('--combine_on_hand_glove_w_hand', default=True, help='Whether to combine glove with hands.')
    args = parser.parse_args()
    

    vis_folder = f'epick_visor_coco_{args.mode}'
    save_folder = f'epick_visor_coco_{args.mode}'
    vis = args.vis
    
    visor_annot_dir = f'{args.epick_visor_store}/annotations'
    visor_img_dir   = f'{args.epick_visor_store}/rgb_frames'
    #
    vis_dir = f'/home/dandans/workspace/HOS/data_preparation/vis/{vis_folder}'
    epick_visor_coco_dir = f'/nfs/turbo/fouheyUnrep/dandans/epick_visor_camera_ready/epick_visor_coco_v7/{save_folder}'

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(epick_visor_coco_dir, exist_ok=True)
    for dir in ['train', 'val', 'test', 'annotations']: 
        if dir in ['train', 'val', 'test']:
            if args.copy_img:
                os.makedirs(f'{epick_visor_coco_dir}/{dir}', exist_ok=True)
        else:
            os.makedirs(f'{epick_visor_coco_dir}/{dir}', exist_ok=True)
            
    # unzip
    if args.unzip_img:
        for split in ['train', 'val', 'test']:
            json_ls = glob.glob(os.path.join(visor_annot_dir, split) + '/*.json')
            json_ls.sort()
            
            for json_path in tqdm(json_ls):
                pid = json_path.split('/')[-1].split('_')[0] # participant id
                vid = json_path.split('/')[-1].split('.')[0]
                img_dir = f'{visor_img_dir}/{split}/{pid}/{vid}'
                if not os.path.exists(img_dir):
                    cmd = f'unzip -o {img_dir}.zip -d {img_dir}'
                    os.system(cmd)
                    
            print(f'Finish unzipping all jsons in {split}')
             
    if args.mode == 'all':
        key_dict, coco_categories = get_coco_category()
    
    
    # for split in ['train', 'val', 'test']:
    for split in args.split:
        img_ls, annot_ls = [], []
        img_id, annot_id = 0, 0
        count_invalid = 0
        count_total_img = 0
        
        json_ls = glob.glob(os.path.join(visor_annot_dir, split) + '/*.json')
        json_ls.sort()
        print(f'#({split} json) = {len(json_ls)}')
        
        # prepare coco format annotation
        if args.num is not None:
            json_ls = json_ls[args.num:]
            
        # fwrite
        fw = open('category.txt', 'a+')
        
        # correction
        correct_dict = json.load(open('correct.json', 'r'))
        for json_path in tqdm(json_ls):
            with open(json_path, 'r') as f:
                info = json.load(f)
                v_annotations = info['video_annotations']
                for i_idx, item in enumerate(v_annotations):
                    entities = item["annotations"]
                    name = item['image']['name']
                    for d_idx, entity in enumerate(entities):
                        # correct missing 'on_which_hand' and 'in_contact_object' for glove
                        if entity['name'] in glove_ls and 'on_which_hand' not in entity:
                            info['video_annotations'][i_idx]['annotations'][d_idx]['on_which_hand'] = None
                            info['video_annotations'][i_idx]['annotations'][d_idx]['in_contact_object'] = None
                            print(f">> no 'on_which_hand' in {item['image']['name']}")
                        # correct typo for 'on_which_hand' for glove
                        if entity['name'] in glove_ls and 'on_which_hand' in entity:
                            if entity['on_which_hand'] is not None and len(entity['on_which_hand'])  == 2:
                                print(f">> wrong 'on_which_hand' in {item['image']['name']}, {entity['on_which_hand']} to ['left hand', 'right hand']")
                                info['video_annotations'][i_idx]['annotations'][d_idx]['on_which_hand'] = ['left hand', 'right hand']
                        # correct contacted obj
                        if name in correct_dict:
                            if entity['name'] in correct_dict[name]:
                                print(name, entity['name'], 'before:', info['video_annotations'][i_idx]['annotations'][d_idx]['in_contact_object'], 'after:', correct_dict[name][entity['name']])
                                info['video_annotations'][i_idx]['annotations'][d_idx]['in_contact_object'] = correct_dict[name][entity['name']]
                                # print(name, entity['name'], 'before:', info['video_annotations'][i_idx]['annotations'][d_idx]['in_contact_object'], 'after:', correct_dict[name][entity['name']])
                                
                        
                                
                
            
            fsave = json_path.replace('annotations', 'annotations_corrected')     
            print(fsave)
            os.makedirs(os.path.split(fsave)[0], exist_ok=True)          
            json.dump(info, open(fsave, 'w'), indent=4)    
            
            
        continue
                        
            
        for json_path in tqdm(json_ls):
            json_path = json_path.replace('annotations', 'annotations_corrected')
            print(split, json_path)
             
            pid = json_path.split('/')[-1].split('_')[0] # participant id
            vid = json_path.split('/')[-1].split('.')[0]
            img_dir = f'{visor_img_dir}/{split}/{pid}/{vid}'
            
                
            with open(json_path, 'r') as f:
                v_annotations = json.load(f)['video_annotations']
                v_annotations = sorted(v_annotations, key=lambda k: k['image']['image_path'])

                sample = Image.open( os.path.join(img_dir, v_annotations[0]['image']['name'] ))
                w, h = int(sample.size[0]), int(sample.size[1])
                    
                for item in v_annotations:
                    # img info
                    count_total_img += 1
                    img_path = item['image']['image_path']
                    img_name = item['image']['name']
                    img_item = {
                        'id': img_id,
                        'file_name': img_name,
                        'height': h,
                        'width': w
                    }
                    
                    # only use images with hands and conclusive hand-object relation annotations in evaluation
                    if split in ['val', 'test']:
                        if check_img_is_invalid(item['annotations']): 
                            count_invalid += 1
                            continue 
                    
                    # prepare annotations
                    entities = item["annotations"]
                    for e_idx, entity in enumerate(entities):
                        # for hands
                        if entity['name'] in ['left hand', 'right hand']: 
                            # set category id
                            if args.mode == 'handonly' or args.mode == 'active':
                                category_id = 1
                            elif args.mode == 'all':
                                category_id = get_category_id(entity['name'], key_dict, fwrite=fw)
                                if category_id is None: 
                                    continue
                            elif args.mode == 'hos':
                                if entity['name'] == 'left hand': 
                                    handside = 0
                                elif entity['name'] == 'right hand': 
                                    handside = 1
                                  
                            # get hand/hand-in-glove mask and in_contact_obj mask
                            img = Image.open(os.path.join(img_dir, img_name))
                            h_masks, h_masks_coco = get_masks(entity['segments'])
                            h_bbox_xywh = get_bbox(h_masks)
                            h_bbox_xyxy = [h_bbox_xywh[0], h_bbox_xywh[1], h_bbox_xywh[0]+h_bbox_xywh[2], h_bbox_xywh[1]+h_bbox_xywh[3]]
                            h_area = get_area(h_masks, height=h, width=w)

                            # check if hand-in-glove, if so combine hand+glove as hand
                            in_contact_object_id = entity['in_contact_object'] 
                            if in_contact_object_id in ['none-of-the-above', 'inconclusive']:                                
                                o_masks = []
                                isincontact = -1       # invalid
                                offset = [-1, -1, -1]  # invalid
                                
                            elif in_contact_object_id in  ['hand-not-in-contact']:
                                o_masks = []
                                isincontact = 0
                                offset = [-1, -1, -1] # invalid
                                
                            else:
                                in_contact_object = [x for x in entities if x['id']== in_contact_object_id][0]

                                
                                # for special glove cases
                                if args.combine_on_hand_glove_w_hand and in_contact_object['name'] in glove_ls:
                                    #! correct typo
                                    if in_contact_object['on_which_hand'] is not None and len(in_contact_object['on_which_hand']) == 2:
                                        in_contact_object['on_which_hand'] = ['left hand', 'right hand']
                                    # for on-hand gloves
                                    if in_contact_object['on_which_hand'] is not None and entity['name'] in in_contact_object['on_which_hand']:
                                        # combine new hand mask = hand masks + on-hand glove
                                        h_masks, h_masks_coco = get_masks(entity['segments'] + in_contact_object['segments'])
                                        h_bbox_xywh = get_bbox(h_masks)
                                        h_bbox_xyxy = [h_bbox_xywh[0], h_bbox_xywh[1], h_bbox_xywh[0]+h_bbox_xywh[2], h_bbox_xywh[1]+h_bbox_xywh[3]]
                                        h_area = get_area(h_masks, height=h, width=w)
                                            
                                        glove_in_contact_object_id = in_contact_object['in_contact_object']
                                        if glove_in_contact_object_id in ['none-of-the-above', 'inconclusive', None]:
                                            o_masks = []
                                            isincontact = -1      
                                            offset = [-1, -1, -1] 
                                            
                                        elif glove_in_contact_object_id in ['glove-not-in-contact']:
                                            o_masks = []
                                            isincontact = 0
                                            offset = [-1, -1, -1] 
                                            
                                        else:
                                            glove_in_contact_object = [x for x in entities if x['id']== glove_in_contact_object_id][0]
                                            #
                                            print(img_path, entity['name'], in_contact_object['name'], glove_in_contact_object['name'])
                                            o_masks, o_masks_coco, o_bbox_xywh, o_bbox_xyxy, o_area, in_contact_object = get_incontact_object(glove_in_contact_object, h_masks, height=h, width=w, image_path=img_path)
                                            isincontact = 1
                                            if o_bbox_xyxy is not None: 
                                                offset = get_offset(h_bbox_xyxy, o_bbox_xyxy)
                                            else:
                                                offset = [-1, -1, -1] 
                                    
                                    # glove, but not on-hand
                                    else:
                                        o_masks, o_masks_coco, o_bbox_xywh, o_bbox_xyxy, o_area, in_contact_object = get_incontact_object(in_contact_object, h_masks, height=h, width=w, image_path=img_path)
                                        isincontact = 1
                                        if o_bbox_xyxy is not None: 
                                            offset = get_offset(h_bbox_xyxy, o_bbox_xyxy)
                                        else:
                                            offset = [-1, -1, -1] 
                                        
                                # not glove 
                                else:
                                    o_masks, o_masks_coco, o_bbox_xywh, o_bbox_xyxy, o_area, in_contact_object = get_incontact_object(in_contact_object, h_masks, height=h, width=w, image_path=img_path)
                                    isincontact = 1
                                    if o_bbox_xyxy is not None: 
                                        offset = get_offset(h_bbox_xyxy, o_bbox_xyxy)
                                    else:
                                        offset = [-1, -1, -1] 
                                      

                            if args.mode in ['handonly', 'active', 'all']:
                                # add hand
                                annot = add_item(
                                        id           = annot_id,
                                        image_id     = img_id, 
                                        category_id  = category_id,    
                                        # 
                                        bbox         = h_bbox_xywh,
                                        area         = h_area,
                                        segmentation = h_masks_coco,
                                        # additional                                    
                                        exhaustive   = entity['exhaustive']
                                  
                                        )

                                annot_ls.append(annot)
                                annot_id += 1
                                
                            elif args.mode == 'hos':
                                # add hand, and in-contact obj if exists
                                if len(o_masks) == 0:
                                    # only add hand
                                    annot = add_item(
                                                id           = annot_id,
                                                image_id     = img_id, 
                                                category_id  = 1,           # hand:1, object:2
                                                # 
                                                bbox         = h_bbox_xywh,
                                                area         = h_area,
                                                segmentation = h_masks_coco,
                                                # additional                                    
                                                exhaustive   = entity['exhaustive'],
                                                handside     = handside,    # left:0, right:1
                                                incontact    = isincontact,
                                                offset       = offset
                                                )

                                    annot_ls.append(annot)
                                    annot_id += 1
                                    
                                    if vis:
                                        img = np.array(img)
                                        cv2.fillPoly(img, h_masks, color_ls[category_id])
                                        cv2.rectangle(img, (h_bbox_xyxy[0], h_bbox_xyxy[1]), (h_bbox_xyxy[2], h_bbox_xyxy[3]), thickness=5, color=color_ls[category_id])
                                        
                                else:
                                    annot = add_item(
                                                id           = annot_id,
                                                image_id     = img_id, 
                                                category_id  = 1,      # hand:1, object:2
                                                #
                                                bbox         = h_bbox_xywh,
                                                area         = h_area,
                                                segmentation = h_masks_coco,
                                                #
                                                exhaustive   = entity['exhaustive'],
                                                handside     = handside,
                                                incontact    = isincontact,
                                                offset       = offset
                                                )
                                    annot_ls.append(annot)
                                    annot_id += 1
                                    
                                    # add an object
                                    annot = add_item(
                                                id           = annot_id,
                                                image_id     = img_id, 
                                                category_id  = 2,      # hand:1, object:2
                                                #
                                                bbox         = o_bbox_xywh,
                                                area         = o_area,
                                                segmentation = o_masks_coco,
                                                # additional
                                                exhaustive   = in_contact_object['exhaustive'],
                                                handside     = -1,
                                                incontact    = -1,
                                                offset       = [-1, -1, -1]
                                                )
                                    annot_ls.append(annot)
                                    annot_id += 1
                                    
                                    
                                    # vis: plot all hands and objs
                                    if vis:
                                        img = np.array(img)
                                        cv2.fillPoly(img, h_masks, color_ls[category_id])
                                        cv2.rectangle(img, (h_bbox_xyxy[0], h_bbox_xyxy[1]), (h_bbox_xyxy[2], h_bbox_xyxy[3]), thickness=5, color=color_ls[category_id])
                                        cv2.fillPoly(img, o_masks, color_ls[category_id])
                                        cv2.rectangle(img, (o_bbox_xyxy[0], o_bbox_xyxy[1]), (o_bbox_xyxy[2], o_bbox_xyxy[3]), thickness=5, color=color_ls[category_id])
                                        
                                            
                                if vis:
                                    img = np.array(img)
                                    vis_path = os.path.join(vis_dir, img_path[:-4]+f'_{e_idx}.jpg')
                                    os.makedirs( os.path.split(vis_path)[0], exist_ok=True)
                                    img_copy = cv2.imread( os.path.join(img_dir, img_name ))
                                    print(img.shape, img_copy.shape)
                                    img_new = cv2.addWeighted(img, 0.4, img_copy, 0.6, 0)
                                    if isinstance(img, np.ndarray):
                                        img = Image.fromarray(img)
                                    img.save(vis_path)
                        
                        
                        # for on-hand gloves   
                        elif  args.combine_on_hand_glove_w_hand and entity['name'] in glove_ls and 'on_which_hand' in entity and entity['on_which_hand'] is not None:    
                            # on-hand glove is already combined with hands, pass
                            
                            # debug
                            if 'on_which_hand' not in  entity:
                                print(f">> {item['image']['image_path']}, {entity['name']} does not have 'on_which_hand' label")
                                pdb.set_trace()
                                continue
                            
                            # the hand in glove exists
                            if check_hand_existence_4_glove(entity, entities):
                                continue
                            
                            
                            # the hand in glove does not exists: count the glove as hand 
                            else:
                                if 'left hand' in entity['on_which_hand']:
                                    handside = 0
                                else:
                                    handside = 1
                                    
                                h_masks, h_masks_coco = get_masks(entity['segments'])
                                h_bbox_xywh = get_bbox(h_masks)
                                h_bbox_xyxy = [h_bbox_xywh[0], h_bbox_xywh[1], h_bbox_xywh[0]+h_bbox_xywh[2], h_bbox_xywh[1]+h_bbox_xywh[3]]
                                h_area = get_area(h_masks, height=h, width=w)

                                # check if hand-in-glove, if so combine hand+glove as hand
                                in_contact_object_id = entity['in_contact_object'] 
                                if in_contact_object_id in ['none-of-the-above', 'inconclusive', None]:                                
                                    o_masks = []
                                    isincontact = -1       # invalid
                                    offset = [-1, -1, -1]  # invalid
                                    
                                elif in_contact_object_id in  ['glove-not-in-contact']:
                                    o_masks = []
                                    isincontact = 0
                                    offset = [-1, -1, -1] # invalid
                                    
                                else:
                                    in_contact_object = [x for x in entities if x['id']== in_contact_object_id][0]

                                    o_masks, o_masks_coco, o_bbox_xywh, o_bbox_xyxy, o_area, in_contact_object = get_incontact_object(in_contact_object, h_masks, height=h, width=w, image_path=img_path)
                                    isincontact = 1
                                    if o_bbox_xyxy is not None: 
                                        offset = get_offset(h_bbox_xyxy, o_bbox_xyxy)
                                    else:
                                        offset = [-1, -1, -1] 
                                        
                                
                                
                                if args.mode in ['handonly']:
                                    continue
                                
                                elif args.mode in ['active', 'all']:
                                    # add hand
                                    annot = add_item(
                                            id           = annot_id,
                                            image_id     = img_id, 
                                            category_id  = category_id,    
                                            # 
                                            bbox         = h_bbox_xywh,
                                            area         = h_area,
                                            segmentation = h_masks_coco,
                                            # additional                                    
                                            exhaustive   = entity['exhaustive']
                                    
                                            )

                                    annot_ls.append(annot)
                                    annot_id += 1
                                    
                                elif args.mode == 'hos':
                                    # add hand, and in-contact obj if exists
                                    if len(o_masks) == 0:
                                        # only add hand
                                        annot = add_item(
                                                    id           = annot_id,
                                                    image_id     = img_id, 
                                                    category_id  = 1,           # hand:1, object:2
                                                    # 
                                                    bbox         = h_bbox_xywh,
                                                    area         = h_area,
                                                    segmentation = h_masks_coco,
                                                    # additional                                    
                                                    exhaustive   = entity['exhaustive'],
                                                    handside     = handside,    # left:0, right:1
                                                    incontact    = isincontact,
                                                    offset       = offset
                                                    )

                                        annot_ls.append(annot)
                                        annot_id += 1
                                        
                                        if vis:
                                            img = np.array(img)
                                            cv2.fillPoly(img, h_masks, color_ls[category_id])
                                            cv2.rectangle(img, (h_bbox_xyxy[0], h_bbox_xyxy[1]), (h_bbox_xyxy[2], h_bbox_xyxy[3]), thickness=5, color=color_ls[category_id])
                                            
                                    else:
                                        annot = add_item(
                                                    id           = annot_id,
                                                    image_id     = img_id, 
                                                    category_id  = 1,      # hand:1, object:2
                                                    #
                                                    bbox         = h_bbox_xywh,
                                                    area         = h_area,
                                                    segmentation = h_masks_coco,
                                                    #
                                                    exhaustive   = entity['exhaustive'],
                                                    handside     = handside,
                                                    incontact    = isincontact,
                                                    offset       = offset
                                                    )
                                        annot_ls.append(annot)
                                        annot_id += 1
                                        
                                        # add an object
                                        annot = add_item(
                                                    id           = annot_id,
                                                    image_id     = img_id, 
                                                    category_id  = 2,      # hand:1, object:2
                                                    #
                                                    bbox         = o_bbox_xywh,
                                                    area         = o_area,
                                                    segmentation = o_masks_coco,
                                                    # additional
                                                    exhaustive   = in_contact_object['exhaustive'],
                                                    handside     = -1,
                                                    incontact    = -1,
                                                    offset       = [-1, -1, -1]
                                                    )
                                        annot_ls.append(annot)
                                        annot_id += 1
                                    
                                    
                                    
                            
                        # for all others (except hands and gloves)
                        else:
                            if args.mode in ['handonly', 'hos']: 
                                continue
                            elif args.mode == 'active':
                                category_id = 2
                            elif args.mode == 'all':
                                category_id = get_category_id(entity['name'], key_dict, fwrite=fw)
                                if category_id is None: 
                                    continue
                                
                                
                            img = Image.open(os.path.join(img_dir, img_name))
                            masks, masks_coco = get_masks(entity['segments'])
                            bbox_xywh = get_bbox(masks)
                            bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0]+bbox_xywh[2], bbox_xywh[1]+bbox_xywh[3]]
                            area = get_area(masks, height=h, width=w)
                            
                            # add
                            annot = add_item(
                                        id           = annot_id,
                                        image_id     = img_id, 
                                        category_id  = category_id,           # hand:1, object:2
                                        # 
                                        bbox         = bbox_xywh,
                                        area         = area,
                                        segmentation = masks_coco,
                                        # additional                                    
                                        exhaustive   = entity['exhaustive']
                                       )

                            annot_ls.append(annot)
                            annot_id += 1
                                
                            if vis:
                                img = np.array(img)
                                cv2.fillPoly(img, masks, color_ls[category_id-1])
                                cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), thickness=5, color=color_ls[category_id-1])
                            
                                img = np.array(img)
                                vis_path = os.path.join(vis_dir, img_path[:-4]+f'_{e_idx}.jpg')
                                os.makedirs( os.path.split(vis_path)[0], exist_ok=True)
                                img_copy = cv2.imread( os.path.join(img_dir, img_name ))
                                print(img.shape, img_copy.shape)
                                img_new = cv2.addWeighted(img, 0.4, img_copy, 0.6, 0)
                                if isinstance(img, np.ndarray):
                                    img = Image.fromarray(img)
                                img.save(vis_path)
                                
                        
                
                    # add image   
                    img_id += 1
                    img_ls.append(img_item)
                    # copy image to corresponding folder in COCO format version (use solflink if there is space limit concern)
                    if args.copy_img:
                        old_img_path = os.path.join(img_dir, img_name)
                        new_img_path = os.path.join(f'{epick_visor_coco_dir}/{split}', img_name)
                        shutil.copy(old_img_path, new_img_path)
                        
                    # pdb.set_trace()
                    
        # category                
        if args.mode == 'handonly':
            categories = [
                {"id": 1, "name": "hand"},
            ]
        elif args.mode in ['active', 'hos']:
            categories = [
                {"id": 1, "name": "hand"},
                {"id": 2, "name": "object"},
            ]
        elif args.mode == 'all':
            categories = coco_categories
            
        # assembly
        coco['info']         = epick_visor_info
        coco['licenses']     = licenses
        coco['categories']   = categories  # 0: hand, 1: object
        coco['images']       = img_ls
        coco['annotations']  = annot_ls

        # save
        f = open(f'{epick_visor_coco_dir}/annotations/{split}.json', 'w')
        json.dump(coco, f, indent=4, cls=NpEncoder)
        
        # print
        print(f'#image = {len(img_ls)}')
        print(f'#annot = {len(annot_ls)}')
        print(f'#(invalid img) = {count_invalid}')
        print(f'#(total img) = {count_total_img}')
