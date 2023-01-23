import numpy as np
import torch
from detectron2.structures import Instances


def hos_postprocessing(predictions):
    '''
    Use predicted offsets to link hand and obj.
    '''
    preds = predictions 
    
    # separate hand, obj instances
    hand_preds = preds[preds.pred_classes == 0]
    obj_preds = preds[preds.pred_classes == 1]
    if len(obj_preds) == 0: return hand_preds
    
    # find incontact obj
    incontact_obj = []
    incontact_obj_ind = []
    modified_hand_preds = []
    for i in range(len(hand_preds)):
        box = hand_preds[i].pred_boxes.tensor.cpu().detach().numpy()[0]
        side = hand_preds[i].pred_handsides.cpu().detach().numpy()[0]
        conta = hand_preds[i].pred_contacts.cpu().detach().numpy()[0]
        offset = hand_preds[i].pred_offsets.cpu().detach().numpy()[0]
        #
        modified_hand_preds.append(hand_preds[i])
        # if incontact
        if int(np.argmax(conta)):
            obj, o_ind = get_incontact_obj(hand_preds[i], offset, obj_preds)
            if isinstance(obj, Instances):
                # same obj box only add once
                if o_ind not in incontact_obj_ind:
                    incontact_obj.append(obj)
                    incontact_obj_ind.append(o_ind)
                
    if len(incontact_obj) > 0:
        ho = incontact_obj + modified_hand_preds
        ho = Instances.cat(ho)
    else:
        if len(modified_hand_preds) > 0:
            ho = Instances.cat(modified_hand_preds)
        else: 
            ho = []
                
    return ho



def combineHO_postprocessing(predictions):
    '''
    Use predicted offsets to link hand and obj, and combine the object to the hand mask.
    '''
    preds = predictions 
    
    # separate hand, obj instances
    hand_preds = preds[preds.pred_classes == 0]
    obj_preds = preds[preds.pred_classes == 1]
    if len(obj_preds) == 0: return hand_preds
    
    # find incontact obj
    incontact_obj = []
    incontact_obj_ind = []
    modified_hand_preds = []
    for i in range(len(hand_preds)):
        box = hand_preds[i].pred_boxes.tensor.cpu().detach().numpy()[0]
        side = hand_preds[i].pred_handsides.cpu().detach().numpy()[0]
        conta = hand_preds[i].pred_contacts.cpu().detach().numpy()[0]
        offset = hand_preds[i].pred_offsets.cpu().detach().numpy()[0]
    
        # if incontact
        if int(np.argmax(conta)):
            obj, o_ind = get_incontact_obj(hand_preds[i], offset, obj_preds)
            if isinstance(obj, Instances):
                # same obj box only add once
                if o_ind not in incontact_obj_ind:
                    incontact_obj.append(obj)
                    incontact_obj_ind.append(o_ind)

                    # new mask, combine hand mask and obj mask
                    combined_masks = torch.logical_or(hand_preds[i].pred_masks, obj.pred_masks)
                    # new box, detectron2.structures.boxes.Boxes, (x1, y1, x2, y2)
                    h_box = hand_preds[i].pred_boxes.tensor
                    o_box = obj.pred_boxes.tensor
                    combined_boxes = torch.zeros_like(h_box)
                    combined_boxes[0,0] = min(h_box[0,0], o_box[0,0])
                    combined_boxes[0,1] = min(h_box[0,1], o_box[0,1])    
                    combined_boxes[0,2] = max(h_box[0,2], o_box[0,2])
                    combined_boxes[0,3] = max(h_box[0,3], o_box[0,3])
                    
                    # update
                    hand_preds[i].pred_masks[0]        = combined_masks
                    hand_preds[i].pred_boxes.tensor[0] = combined_boxes
                    
        # add hand pred to list     
        modified_hand_preds.append(hand_preds[i])
    
    if len(modified_hand_preds) > 0:
        combined_ho = Instances.cat(modified_hand_preds)
    else: 
        combined_ho = []
  
    
    return combined_ho



def get_offset(h_bbox_xyxy, o_bbox_xyxy):
    '''
    Calculate offset from hand to object bbox, using xyxy bbox annotation.
    '''
    h_center = [int((h_bbox_xyxy[0] + h_bbox_xyxy[2]) / 2), int((h_bbox_xyxy[1] + h_bbox_xyxy[3]) / 2)]
    o_center = [int((o_bbox_xyxy[0] + o_bbox_xyxy[2]) / 2), int((o_bbox_xyxy[1] + o_bbox_xyxy[3]) / 2)]
    # offset: [vx, vy, magnitute], 
    scalar = 1000 
    vec = np.array([o_center[0]-h_center[0], o_center[1]-h_center[1]]) / scalar
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm
    offset = [unit_vec[0], unit_vec[1], norm]
    return offset    
    
    pdb.set_trace()

def get_incontact_obj(h_box, offset, obj_preds):
    
    '''
    Find in-contact object for hand that is predicted as in-contact.
    '''
    h_center = get_center(h_box)
    scalar = 1000
    offset_vec = [ offset[0]*offset[2]*scalar, offset[1]*offset[2]*scalar ] 
    pred_o_center = [h_center[0]+offset_vec[0], h_center[1]+offset_vec[1]]
    
    # choose from obj_preds
    dist_ls = []
    for i in range(len(obj_preds)):
        o_center = get_center(obj_preds[i])
        dist = np.linalg.norm(np.array(o_center) - np.array(pred_o_center))
        dist_ls.append(dist)
    
    if len(dist_ls) == 0: 
        return [], None
    else:
        o_ind = np.argmin(np.array(dist_ls))
        return obj_preds[int(o_ind)], o_ind


def get_center(box):
    '''
    '''
    box = box.pred_boxes.tensor.cpu().detach().numpy()[0]
    x0, y0, x1, y1 = box
    center = [int((x0+x1)/2), int((y0+y1)/2)]
    return center
