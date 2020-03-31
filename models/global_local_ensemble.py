from .resnet import resnet50, resnet18
from .fpn_semantci_flow import get_fpn_sf_global, get_fpn_sf_local
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def get_fpn_global(num_classes, mode, pretrained=False, path=None):
    model = FPN_G(num_classes, mode)
    if pretrained and path:
        state = model.state_dict()
        state.update(torch.load(path))
        model.load_state_dict(state)
    
    return model

def get_fpn_local(num_classes, mode, pretrained=False, path=None):
    model = FPN_L(num_classes, mode)
    if pretrained and path:
        state = model.state_dict()
        state.update(torch.load(path))
        model.load_state_dict(state)
    
    return model


class FPN_G(nn.Module):
    def __init__(self, num_classes, mode):
        super(FPN_G, self).__init__()
        self.mode = mode
        self.backbone = resnet50(True)
        self.fpn = get_fpn_sf_global(num_classes, mode=self.mode)
    
    def forward(self, img):
        c2, c3, c4, c5 = self.backbone(img)
        output = self.fpn(c2, c3, c4, c5)

        return output


class FPN_L(nn.Module):
    def __init__(self, num_classes, mode):
        super(FPN_L, self).__init__()
        self.mode = mode
        self.backbone = resnet18(True)
        self.fpn = get_fpn_sf_local(num_classes, mode=self.mode)
    
    def forward(self, img):
        c2, c3, c4, c5 = self.backbone(img)
        output = self.fpn(c2, c3, c4, c5)

        return output


class FPN_GL(nn.Module):
    def __init__(self, num_classes, path_g=None, path_l=None):
        super(FPN_GL, self).__init__()
        self.mode = 'SemanticFlow' # 'Bilinear'
        # fpn module
        self.fpn_global = get_fpn_global(num_classes, mode=self.mode, pretrained=True, path=path_g)
        self.fpn_local = get_fpn_local(num_classes, mode=self.mode, pretrained=True,  path=path_l)
        # dual fam for global and local branch
        self.dual_fam1 = DualFAM(features=256)
        self.dual_fam2 = DualFAM(features=256)
        self.dual_fam3 = DualFAM(features=256)
        self.dual_fam4 = DualFAM(features=256)
        # smooth layer for feature fusion
        self.smooth1_g = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2_g = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3_g = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth4_g = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1_l = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2_l = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3_l = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        self.smooth4_l = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0)
        
        # ensemble
        self.ensemble_conv = nn.Conv2d(256*2, num_classes, kernel_size=3, stride=1, padding=1)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(reduce=False)
        #init
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.ensmble_conv.weight, mean=0, std=0.01)
        for m in self.dual_fam1.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): 
                nn.init.constant_(m.bias, 0)
        for m in self.dual_fam2.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): 
                nn.init.constant_(m.bias, 0)
        for m in self.dual_fam3.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): 
                nn.init.constant_(m.bias, 0)
        for m in self.dual_fam4.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): 
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img_g, img_l, target=None):
        # _, h, w = target.size() # x8
        c2_g, c3_g, c4_g, c5_g = self.fpn_global.backbone(img_g)
        c2_l, c3_l, c4_l, c5_l = self.fpn_global.backbone(img_l)

        # calculate feature fusion weight
        with torch.no_grad():
            output_l = self.fpn_local.fpn.forward(c2_l, c3_l, c4_l, c5_l) # x8
            _, _, h, w = output_l.size()
            output_g = self.fpn_global.fpn.forward(c2_g, c3_g, c4_g, c5_g) # x16
            output_g = F.interpolate(output_g, size=(h, w), mode='bilinear')
            
            if target:
                loss_g = self.ce(output_g, target)
                loss_l = self.ce(output_l, target)
            else:
                loss_g = nn.mean(nn.LogSoftmax(output_g, dim=1), dim=1)
                loss_l = nn.mean(nn.LogSoftmax(output_l, dim=1), dim=1)
            weight_g = nn.Sigmoid(loss_l-loss_g) # x8
            weight_l = nn.Sigmoid(loss_g-loss_l) # x8
        
        # get lateral features from two branch
        ps_g = self.fpn_global.fpn.get_fusion_feature_pyramid(c2_g, c3_g, c4_g, c5_g)
        ps_l = self.fpn_local.fpn.get_fusion_feature_pyramid(c2_l, c3_l, c4_l, c5_l)
        
        # align and fuse features
        ts_g, ts_l = self.align_fuse_features(ps_g, ps_l, weight_g, weight_l)

        # classify
        output_g, ensemble_g = self.fpn_global.fpn.forward_with_lateral(ts_g)
        output_l, ensemble_l = self.fpn_local.fpn.forward_with_lateral(ts_l)
        output_g = F.interpolate(output_g, size=(h, w), mode='bilinear')
        ensemble_g = F.interpolate(ensemble_g, size=(h, w), mode='bilinear')

        mse = self.mse(output_g, output_l)
        
        # ensemble
        output = self.ensemble_conv(torch.cat([ensemble_g, ensemble_l], dim=1))

        return output, output_g, output_l, mse
        
    def align_fuse_features(ps_g, ps_l, weight_g, weight_l):
        # align lateral features
        p2_g, p2_l = self.dual_fam1(ps_g[0], ps_l[0])
        p3_g, p3_l = self.dual_fam1(ps_g[1], ps_l[1])
        p4_g, p4_l = self.dual_fam1(ps_g[2], ps_l[2])
        p5_g, p5_l = self.dual_fam1(ps_g[3], ps_l[3])

        # reweight fusion feature
        p2_g = p2_g.mul(weight_g); p2_l = p2_l.mul(weight_l)
        p3_g = p3_g.mul(weight_g); p3_l = p3_l.mul(weight_l)
        p4_g = p4_g.mul(weight_g); p4_l = p4_l.mul(weight_l)
        p5_g = p5_g.mul(weight_g); p5_l = p5_l.mul(weight_l)

        # fuse lateral features
        t2_g = self.smooth1_g(torch.cat([ps_g[0], p2_l])); t2_l = self.smooth1_l(torch.cat([ps_l[0], p2_g]))
        t3_g = self.smooth2_g(torch.cat([ps_g[1], p3_l])); t3_l = self.smooth2_l(torch.cat([ps_l[1], p3_g]))
        t4_g = self.smooth3_g(torch.cat([ps_g[2], p4_l])); t4_l = self.smooth3_l(torch.cat([ps_l[2], p4_g]))
        t5_g = self.smooth4_g(torch.cat([ps_g[3], p5_l])); t5_l = self.smooth4_l(torch.cat([ps_l[3], p5_g]))

        return [t2_g, t3_g, t4_g, t5_g], [t2_l, t3_l, t4_l, t5_l]



class DualFAM(nn.Module):
    def __init__(self, features=256):
        super().__init__()
        self.smooth1_x = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.smooth1_y = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.flow_x = nn.Sequential(nn.Conv2d(features*2, features, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(features),
                                nn.ReLU(True),
                                nn.Conv2d(features, 2, kernel_size=3, stride=1, padding=1))
        self.flow_y = nn.Sequential(nn.Conv2d(features*2, features, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(features),
                                nn.ReLU(True),
                                nn.Conv2d(features, 2, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, h_x, w_x = x.size()
        _, _, h_y, w_y = y.size()
        x_smooth = self.smooth_x(x)
        y_smooth = self.smooth_y(y)
        x_resize = F.interpolate(x, size=(h_y, w_y), mode='bilinear', align_corners=True)
        y_resize = F.interpolate(y, size=(h_x, w_x), mode='bilinear', align_corners=True)
        flow_x = self.flow_x(torch.cat([x_smooth, y_resize], dim=1))
        flow_y = self.flow_y(toch.cat([y_smooth, x_resize], dim=1))
        x_warp = self.stn(x, flow_x)
        y_warp = self.stn(y, flow_y)

        return x_warp, y_warp
    
    def stn(self, x, flow):
        _, _, H, W = flow.size()
        grid_h, grid_w = torch.meshgrid(torch.range(0, H-1)/H, torch.range(0, W-1)/W)
        flow[:, 0] += grid_h
        flow[:, 1] += grid_w
        flow = flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, flow)