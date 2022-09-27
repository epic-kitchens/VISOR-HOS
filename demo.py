from detectron2.utils.logger import setup_logger
setup_logger()

import os, pdb, random, argparse, cv2, glob
random.seed(0)
import numpy as np
from tqdm import tqdm

import torch
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode , Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

# import PointRend project
from detectron2.projects import point_rend
from hos.data.datasets.epick import register_epick_instances
from hos.data.hos_datasetmapper import HOSMapper
from hos.visualization.v import Visualizer as HOS_Visualizer

# register epick visor dataset
version = 'datasets/epick_visor_coco_hos'
register_epick_instances("epick_visor_2022_val_hos", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_hos").thing_classes = ["hand", "object"]
epick_visor_metadata = MetadataCatalog.get("epick_visor_2022_val_hos")


def run(mode=None, test_img_ls=None, out_dir=None, pointrend_cfg=None, model_weight=None, use_postprocess=False):
    '''
    Run demo on images.
    '''
    # config
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(pointrend_cfg)
    
    # set mode, e.g. number of classes
    if mode == 'hos' or mode == 'active':
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_weight
    predictor = DefaultPredictor(cfg)
    
    # output dir
    if use_postprocess:
        out_dir = out_dir + '_postprocess'
        os.makedirs(out_dir, exist_ok=True)
        mode += '_postprocess'
        

    for img_path in tqdm(test_img_ls):

        im = cv2.imread(img_path)
        out_path = os.path.join(out_dir, img_path.split('/')[-1].split('.')[0]+'_pred.jpg')
        outputs = predictor(im)
        
        if mode == 'all':
            v = Visualizer(im[:, :, ::-1], epick_visor_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        else:
            v = HOS_Visualizer(im[:, :, ::-1], epick_visor_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
       
        if use_postprocess:
            outputs = hos_postprocessing(outputs)
        # pdb.set_trace()
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        cv2.imwrite(out_path, point_rend_result[:, :, ::-1])
        
    
      

def hos_postprocessing(predictions):
    '''
    Use predicted offsets to associate hand and its in-contact obj.
    '''
    preds = predictions['instances'].to("cpu")
    if len(preds) == 0: return predictions
    # separate hand, obj preds
    hand_preds = preds[preds.pred_classes == 0]
    obj_preds = preds[preds.pred_classes == 1]
    
    if len(obj_preds) == 0: return {'instances':hand_preds}
    
    # find incontact obj
    incontact_obj = []
    updated_hand_preds = []
    for i in range(len(hand_preds)):
        box = hand_preds[i].pred_boxes.tensor.cpu().detach().numpy()[0]
        side = hand_preds[i].pred_handsides.cpu().detach().numpy()[0]
        contact = hand_preds[i].pred_contacts.cpu().detach().numpy()[0]
        offset = hand_preds[i].pred_offsets.cpu().detach().numpy()[0]
        # if incontact
        if int(np.argmax(contact)):
            obj = get_incontact_obj(hand_preds[i], offset, obj_preds)
            if isinstance(obj, Instances):
                incontact_obj.append(obj)
                new = Instances(hand_preds[i].image_size)
                for field in hand_preds[i]._fields:
                    if field == 'pred_offsets':
                        new.set(field, torch.Tensor([get_offset(box, obj.pred_boxes.tensor.cpu().detach().numpy()[0])])  )
                    else:
                        new.set(field, hand_preds[i].get(field))
                updated_hand_preds.append(new)
               
        else:
            updated_hand_preds.append(hand_preds[i])
            
    if len(incontact_obj) > 0:
        incontact_obj.extend(updated_hand_preds)
        ho = Instances.cat(incontact_obj)
    else:
        if len(updated_hand_preds) > 0:
            ho = Instances.cat(updated_hand_preds)
        else:
            ho = Instances( preds[0].image_size)
        
    return {'instances': ho}


def get_offset(h_bbox_xyxy, o_bbox_xyxy):
    '''
    Calculate offset from hand to object bbox, using xyxy bbox annotation.
    '''
    h_center = [int((h_bbox_xyxy[0] + h_bbox_xyxy[2]) / 2), int((h_bbox_xyxy[1] + h_bbox_xyxy[3]) / 2)]
    o_center = [int((o_bbox_xyxy[0] + o_bbox_xyxy[2]) / 2), int((o_bbox_xyxy[1] + o_bbox_xyxy[3]) / 2)]
    # offset: [vx, vy, magnitute]
    scalar = 1000 
    vec = np.array([o_center[0]-h_center[0], o_center[1]-h_center[1]]) / scalar
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm
    offset = [unit_vec[0], unit_vec[1], norm]
    return offset    
    

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
        return []
    else:
        o_ind = np.argmin(np.array(dist_ls))
        return obj_preds[int(o_ind)]


def get_center(box):
    box = box.pred_boxes.tensor.cpu().detach().numpy()[0]
    x0, y0, x1, y1 = box
    center = [int((x0+x1)/2), int((y0+y1)/2)]
    return center


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, default='./inputs')
    parser.add_argument('--output_dir', type=str, required=True, help='./outputs')
    args = parser.parse_args()
    
    
    # get images
    img_ls = glob.glob(f'{args.input_dir}/*')
    test_img_ls = []
    for img_path in img_ls:
        if os.path.exists(img_path) and img_path.endswith(('.jpg', '.JPEG', '.png', '.PNG')):
            test_img_ls.append(img_path)
    print(f'Got {len(test_img_ls)} images in total.')
    test_img_ls.sort()
    
    
    # run inference
    task_ls =  ['hos', 'active']
    for task in task_ls :
        
        print(f'Running {task} task...')
        out_dir    = f'{args.output_dir}/{task}'
        os.makedirs(out_dir, exist_ok=True)
        if task == 'hos':
            pointrend_cfg = "./configs/hos/hos_pointrend_rcnn_R_50_FPN_1x_trainset.yaml"
            epick_model = f'./checkpoints/model_final_hos.pth'
            run(task, test_img_ls, out_dir, pointrend_cfg, epick_model, use_postprocess=True)
        elif task == 'active':
            pointrend_cfg = "./configs/active/active_pointrend_rcnn_R_50_FPN_1x_trainset.yaml"
            epick_model = f'./checkpoints/model_final_active.pth'

        run(task, test_img_ls, out_dir, pointrend_cfg, epick_model, use_postprocess=False)
        
        
        
    