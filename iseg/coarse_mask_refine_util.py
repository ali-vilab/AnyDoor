"""MobileNet and MobileNetV2."""
'''
Code adopted from https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/models/backbones/mobilenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============  Basic Blocks  ============

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels, 3, stride, 1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ============  Backbone  ============

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        output_stride = 8
        self.multiplier = 1
        if output_stride == 32:
            dilations = [1, 1]
        elif output_stride == 16:
            dilations = [1, 2]
        elif output_stride == 8:
            dilations = [2, 4]
        else:
            raise NotImplementedError
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        # building first layer
        input_channels = int(32 * self.multiplier) if self.multiplier > 1.0 else 32
        # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = _ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)

        # building inverted residual blocks
        self.planes = input_channels
        self.block1 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[0:1],
                                       norm_layer=norm_layer)
        self.block2 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[1:2],
                                       norm_layer=norm_layer)
        self.block3 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[2:3],
                                       norm_layer=norm_layer)
        self.block4 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[3:5],
                                       dilations[0], norm_layer=norm_layer)
        self.block5 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[5:],
                                       dilations[1], norm_layer=norm_layer)
        self.last_inp_channels = self.planes

        # building last several layers
        # features = list()
        # features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6=True, norm_layer=norm_layer))
        # features.append(nn.AdaptiveAvgPool2d(1))
        # self.features = nn.Sequential(*features)
        #
        # self.classifier = nn.Sequential(
        #     nn.Dropout2d(0.2),
        #     nn.Linear(last_channels, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, inverted_residual_setting, dilation=1, norm_layer=nn.BatchNorm2d):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(block(planes, out_channels, stride, t, dilation, norm_layer))
            planes = out_channels
            for i in range(n - 1):
                features.append(block(planes, out_channels, 1, t, norm_layer=norm_layer))
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def forward(self, x, side_feature):
        x = self.conv1(x)
        x = x + side_feature
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.block5(c3)
        # x = self.features(x)
        # x = self.classifier(x.view(x.size(0), x.size(1)))
        return c1, c2, c3, c4

def mobilenet_v2(norm_layer=nn.BatchNorm2d):
    return MobileNetV2(norm_layer=norm_layer)



# ============  Segmentor  ============

class LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(LRASPP, self).__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  
        return x



class MobileSeg(nn.Module):
    def __init__(self, nclass=1, **kwargs):
        super(MobileSeg, self).__init__()
        self.backbone = mobilenet_v2()
        self.lraspp = LRASPP(320,128)
        self.fusion_conv1 = nn.Conv2d(128,16,1,1,0)
        self.fusion_conv2 = nn.Conv2d(24,16,1,1,0)
        self.head = nn.Conv2d(16,nclass,1,1,0)
        self.aux_head = nn.Conv2d(16,nclass,1,1,0)

    def forward(self, x, side_feature):
        x4, _, _, x8 = self.backbone(x, side_feature)
        x8 = self.lraspp(x8)
        x8 = F.interpolate(x8, x4.size()[2:], mode='bilinear', align_corners=True)
        x8 = self.fusion_conv1(x8)
        pred_aux = self.aux_head(x8)

        x4 = self.fusion_conv2(x4)
        x = x4 + x8
        pred = self.head(x)
        return pred, pred_aux, x

    def load_pretrained_weights(self, path_to_weights= ' '):    
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(backbone_state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Loading Mobilnet V2')
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict, strict= False)




class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


# ============ Interactive Segmentor  ============

class BaselineModel(nn.Module):
    def __init__(self, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.feature_extractor = MobileSeg()
        side_feature_ch = 32
        mt_layers = [
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=16, out_channels=side_feature_ch, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)

    
    def backbone_forward(self, image, coord_features=None):
        mask, mask_aux, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask_aux, 'feature': feature}


    def prepare_input(self, image):
        prev_mask = torch.zeros_like(image)[:,:1,:,:]
        return image, prev_mask

    def forward(self, image, coarse_mask):
        image, prev_mask = self.prepare_input(image)
        coord_features = torch.cat((prev_mask, coarse_mask, coarse_mask * 0.0), dim=1)
        click_map = coord_features[:,1:,:,:]

        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features)
        
        pred = nn.functional.interpolate(
                                outputs['instances'], 
                                size=image.size()[2:],
                                mode='bilinear', align_corners=True
                                )

        outputs['instances'] = torch.sigmoid(pred)
        return outputs



