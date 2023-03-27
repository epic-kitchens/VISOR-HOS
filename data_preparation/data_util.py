import json, cv2, csv, pdb
import numpy as np
from scipy import ndimage


class NpEncoder(json.JSONEncoder):
    """
    Numpy encoder for json.dump().
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def check_existence(name, entites):
    """
    Check if an entity exists in the list.
    """
    for item in entites:
        if item['name'] == name:
            return True
    return False
        

def check_hand_existence_4_glove(entity, entites):
    '''
    Check only the on-hand glove exist but the hand is not visible.
    '''
    # check the glove is on hand
    hand_exists = False
    if entity['on_which_hand'] is not None:
        for hand in entity['on_which_hand']:
            # check the hand is not visible
            if check_existence(hand, entites):
                hand_exists =  True
    return hand_exists


def check_img_is_invalid(annotations):
    """
    Current image has hands with valid hand-object relation annotation.
    """
    isInValid = False
    glove_ls = ['oven glove', 'gloves', 'rubber glove', 'left glove', 'right glove', 'glove']
    for item in annotations:
        if item['name'] in ['left hand', 'right hand']:
            # hand is inconclusive
            if item['in_contact_object'] in ['none-of-the-above', 'inconclusive']:
                isInValid = True
        if item['name'] in glove_ls:
            # glove is on-hand & inconclusive
            if 'on_which_hand' in item and item['on_which_hand'] is not None and item['in_contact_object'] in ['none-of-the-above', 'inconclusive']:
                isInValid = True
            
            if 'on_which_hand' in item and item['on_which_hand'] is not None and len(item['on_which_hand'])==2:
                isInValid = True
    
    return isInValid


def get_iou_overlap(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return [0.0, 0.0]

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    overlap = intersection_area / float(bbox1_area)
    assert iou >= 0.0
    assert iou <= 1.0
    assert overlap >= 0.0
    assert overlap <= 1.0
    # TODO: check return
    return [iou, overlap]


def get_bbox(masks):
    '''
    Get bbox for object masks (1 object may have 1> components).
    
    Returns:
        bbox: [x, y, height, width]
    '''
    g_xmin, g_ymin, g_xmax, g_ymax = 10000, 10000, 0, 0
    for mask in masks:
        if len(mask) == 0: continue
        mask = np.array(mask)
        xmin, xmax = np.min(mask[:,0]), np.max(mask[:,0])
        ymin, ymax = np.min(mask[:,1]), np.max(mask[:,1])

        g_xmin = min(g_xmin, xmin)
        g_xmax = max(g_xmax, xmax)
        g_ymin = min(g_ymin, ymin)
        g_ymax = max(g_ymax, ymax)

    bbox = [int(g_xmin), int(g_ymin), int(g_xmax - g_xmin), int(g_ymax - g_ymin)]
    return bbox


def get_area(mask, height, width):
    '''
    Get mask area by plotting on a binary mask and counting 1s.
    '''
    im = np.zeros([height, width],dtype=np.uint8)
    cv2.fillPoly(im, mask, color=(1, 1, 1))
    area = np.count_nonzero(im)
    return area


def get_offset(h_bbox_xyxy, o_bbox_xyxy):
    '''
    Calculate offset from hand to object bbox center, using xyxy bbox annotation.
    
    Returns:
        offset: [unit_vector[0], unit_vector[1], magnitude]
    '''
    h_center = [int((h_bbox_xyxy[0] + h_bbox_xyxy[2]) / 2), int((h_bbox_xyxy[1] + h_bbox_xyxy[3]) / 2)]
    o_center = [int((o_bbox_xyxy[0] + o_bbox_xyxy[2]) / 2), int((o_bbox_xyxy[1] + o_bbox_xyxy[3]) / 2)]

    scalar = 1000 #TODO: the scalar needs testing
    vec = np.array([o_center[0]-h_center[0], o_center[1]-h_center[1]]) / scalar
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm
    
    offset = [unit_vec[0], unit_vec[1], norm]
    return offset    


def get_masks(annot):
    """
    Generate masks in certain formats.
    
    Return:
        
    """ 
    masks_coco = []
    masks_clean = []
    for mask in annot:
        if len(mask) == 0: continue
        mask_ls = list(np.concatenate(mask))
        maskRounded = [round(x,2) for x in mask_ls] 
        if len(maskRounded) < 6: # too few points
            print(maskRounded)
            continue
        mask = np.array(mask, dtype=np.int32)
        masks_coco.append(maskRounded)
        masks_clean.append(mask)
    
    # pdb.set_trace()
    return masks_clean, masks_coco


def combine_hand_object(h_masks, o_masks, h_masks_coco, o_masks_coco):
    if h_masks is not None and o_masks is not None:
        h_masks.extend(o_masks)
        h_masks_coco.extend(o_masks_coco)
        return h_masks, h_masks_coco
    else:
        pdb.set_trace()
        return None

    
def get_incontact_object(in_contact_object=None, h_masks=None, height=None, width=None, image_path=None):
    '''
    Get in-contact object for each hand, only the incontact component of the object.
    '''
    
    # only the in-contact component of the object
    # for in_contact_object in in_contact_object_ls:
    o_masks, o_masks_coco = get_masks(in_contact_object['segments'])
    is_valid, incontact_o_masks, incontact_o_masks_coco = get_incontact_component(o_masks, o_masks_coco, h_masks, height, width)
    
    if is_valid:
        o_bbox_xywh = get_bbox(incontact_o_masks)
        o_bbox_xyxy = [o_bbox_xywh[0], o_bbox_xywh[1], o_bbox_xywh[0]+o_bbox_xywh[2], o_bbox_xywh[1]+o_bbox_xywh[3]]
        o_area = get_area(incontact_o_masks, height=height, width=width)
        return incontact_o_masks, incontact_o_masks_coco, o_bbox_xywh, o_bbox_xyxy, o_area, in_contact_object
    else:
        print(f'*** This might be there are >1 instances of the same name')
        # pdb.set_trace()
        # continue
        return [], [], None, None, None, None
    
    
def get_incontact_component(o_masks, o_masks_coco, h_masks, height=None, width=None):
    
    '''
    Get only the incontact component of the object.
    
    special case:
    *** Weird!, overlap not consistent, P12_101/0028_P12_101_Part_001/0028_P12_101_seq_00006/0028_frame_0000003059/0028_P12_101_frame_0000003059.jp
    '''
    
    # hand binary mask
    hand_bin = np.zeros([height, width],dtype=np.uint8)
    cv2.fillPoly(hand_bin, h_masks, (1, 1, 1))
    hand_bin = dilate(hand_bin)
    
    incontact_o_masks = []
    incontact_o_masks_coco = []
    for o_mask, o_mask_coco in zip(o_masks, o_masks_coco):
        # obj binary mask
        obj_bin = np.zeros([height, width],dtype=np.uint8)
        cv2.fillPoly(obj_bin, [o_mask], (1,1,1))
        obj_bin = dilate(obj_bin) # dilate mask
        
        # if overlap
        overlap_score = np.sum( np.bitwise_and(hand_bin, obj_bin) )   #overlap pixel
        if overlap_score > 0:
            incontact_o_masks.append(o_mask)
            incontact_o_masks_coco.append(o_mask_coco) 
    
    if len(incontact_o_masks) > 0:  
        return True, incontact_o_masks, incontact_o_masks_coco
    else:
        return False, incontact_o_masks, incontact_o_masks_coco
    

def dilate(mask, iterations=5):
    mask = ndimage.binary_dilation(mask, iterations=iterations).astype(mask.dtype)
    return mask



def add_item(image_id=None, 
              category_id=None,
              id=None,
              bbox=None,
              area=None,
              segmentation=None,
              iscrowd=0,
              exhaustive=None,
              handside=None,
              incontact=None,
              offset=None
              ):
    item = {}
    item['id'] = id
    item['image_id'] = image_id
    item['category_id'] = category_id
    #
    item['bbox'] = bbox
    item['area'] = area
    item['segmentation'] = segmentation
    item['iscrowd'] = iscrowd
    # additional
    item['exhaustive'] = exhaustive
    item['handside'] = handside
    item['isincontact'] = incontact
    item['offset'] = offset

    return item



# --- 
def transfer_noun(noun):
    if ':' not in noun: 
        return noun
    List = noun.split(':')
    return ' '.join(List[1:]) + ' ' + List[0]

def get_coco_category(csv_path='./EPIC_100_noun_classes_v2.csv'):
    key_dict = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            key_idx = int(row['id']) + 1
            key = transfer_noun( row['key'] )
            instances = [x.strip()[1:-1] for x in row['instances'][1:-1].split(',')]
            inst_ls = [transfer_noun(inst) for inst in instances]
            category = row['category']
            
            key_dict[key_idx] = {}
            key_dict[key_idx]['key'] = key
            key_dict[key_idx]['instances'] = inst_ls
            key_dict[key_idx]['category'] = category
            
    coco_categories = [ {'id':kind, 'name':kval['key']} for kind, kval in key_dict.items()]
    return key_dict, coco_categories

def get_category_id(entity_name, key_dict, fwrite=None):
    entity_name = entity_name.strip()
    entity_name = entity_name.lower()
    for kind, kval in key_dict.items():
        if entity_name in kval['instances']:
            return kind
        
    print(f"*** Error: entity name not in csv: {entity_name}")
    if fwrite is not None:
        fwrite.write(f'{entity_name} not found\n')
    return None
