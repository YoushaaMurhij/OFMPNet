
from torchmetrics import MeanMetric

class OGMFlowMetrics(object):
    def __init__(self, device, no_warp=False):
        # super().__init__()
        self.observed_auc = MeanMetric().to(device)
        self.occluded_auc = MeanMetric().to(device)

        self.observed_iou = MeanMetric().to(device)
        self.occluded_iou = MeanMetric().to(device)
        
        self.flow_epe = MeanMetric().to(device)
        self.no_warp = no_warp

        if not no_warp:
            self.flow_ogm_auc = MeanMetric().to(device)
            self.flow_ogm_iou = MeanMetric().to(device)

    
    
    def update(self, metrics):
        self.observed_auc.update(metrics.vehicles_observed_auc)
        self.occluded_auc.update(metrics.vehicles_occluded_auc)

        self.observed_iou.update(metrics.vehicles_observed_iou)
        self.occluded_iou.update(metrics.vehicles_occluded_iou)

        self.flow_epe.update(metrics.vehicles_flow_epe)
        if not self.no_warp:
            self.flow_ogm_auc.update(metrics.vehicles_flow_warped_occupancy_auc)
            self.flow_ogm_iou.update(metrics.vehicles_flow_warped_occupancy_iou)
    
    def compute(self):
        res_dict={}
        res_dict['observed_auc'] = self.observed_auc.compute()
        res_dict['occluded_auc'] = self.occluded_auc.compute()

        res_dict['observed_iou'] = self.observed_iou.compute()
        res_dict['occluded_iou'] = self.occluded_iou.compute()

        res_dict['flow_epe'] = self.flow_epe.compute()
        if not self.no_warp:
            res_dict['flow_ogm_auc'] = self.flow_ogm_auc.compute()
            res_dict['flow_ogm_iou'] = self.flow_ogm_iou.compute()

        return res_dict
    
def print_metrics(res_dict, no_warp=False):
    # print(f'\n |obs-AUC: {res_dict['observed_auc']}')
    if no_warp:
       print(f"""\n |obs-AUC: {res_dict['observed_auc']}|occ-AUC: {res_dict['occluded_auc']}
            |obs-IOU: {res_dict['observed_iou']}|occ-IOU: {res_dict['occluded_iou']}
            | Flow-EPE: {res_dict['flow_epe']}|""", flush=True)
    else: 
        print(f"""\n |obs-AUC: {res_dict['observed_auc']}|occ-AUC: {res_dict['occluded_auc']}
                |obs-IOU: {res_dict['observed_iou']}|occ-IOU: {res_dict['occluded_iou']}
                | Flow-EPE: {res_dict['flow_epe']}
                |FlowOGM_AUC: {res_dict['flow_ogm_auc']} |FlowOGM_IOU: {res_dict['flow_ogm_iou']} |""", flush=True)
        