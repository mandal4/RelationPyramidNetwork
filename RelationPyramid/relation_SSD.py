import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
from layers.box_utils import match
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
cls_r_prob = pickle.load(open('data/graph/VOC_graph_r.pkl', 'rb'))
cls_r_prob = np.float32(cls_r_prob)
cls_r_prob=nn.Parameter(torch.from_numpy(cls_r_prob))

class relation_SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, rel, upscalingConv, bufConv):
        super(relation_SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = Variable(self.priorbox.forward(),requires_grad=False)
        #torch.no_grad()

        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.rel_transform = nn.ModuleList(rel)
        self.upscaling_conv= nn.ModuleList(upscalingConv)
        self.buf_conv = nn.ModuleList(bufConv)
        self.getEnhancedSource=get_enhanced_source()
        #self.getEnhancedSource=self.getEnhancedSource.cuda()

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    def forward(self, x, targets):


        sources = list()
        loc = list()
        conf = list()
        #self.targets=targets
        # apply vgg up to conv4_3 relu
        #x.size(): [1,64,300,300]
        for k in range(23):
            x = self.vgg[k](x)
        # x.size(): [1,512,38,38]

        s = self.L2Norm(x)
        # s.size(): [1,512,38,38]

        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # x.size(): [1,1024,19,19]
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                #x.size(): [1,512,10,10],[1,256,5,5],[1,256,3,3],[1,256,1,1]

        sources.reverse()
        #time_list = list()
        #time_list.append(time.time())
        # apply multibox head to source layers
        stage=0

        for (x, l, c) in zip(sources, self.loc, self.conf):  #self.loc, self.conf: head # all for len '6'
            if (stage==0 or stage==5 or stage==1 or stage ==3 ):
                conf_pred_prev = c(x).permute(0, 2, 3, 1).contiguous()
                loc_pred_prev = l(x).permute(0, 2, 3, 1).contiguous()

                loc.append(loc_pred_prev)
                conf.append(conf_pred_prev)
                x_prev=x
             #   time_list.append(time.time())

            else:
                #print('Current step: {}'.format(stage))
                fusedBuffer, separatedSource = self.getEnhancedSource(self.priors, targets, self.num_classes,x_prev, conf_pred_prev, loc_pred_prev,stage)
                fused_feature = self.rel_transform[stage-1](fusedBuffer)
                reasoned_output = torch.cat((separatedSource, fused_feature), dim=2)
                enhanced_x_prev = reasoned_output.view(reasoned_output.size()[0],x_prev.size()[2],x_prev.size()[3],reasoned_output.size()[2])
                enhanced_x_prev = enhanced_x_prev.permute(0,3,2,1)
                x_prev = self.buf_conv[stage-1](x) + self.upscaling_conv[stage-1](enhanced_x_prev)

                loc_pred_prev = l(x_prev).permute(0, 2, 3, 1).contiguous()
                conf_pred_prev = c(x_prev).permute(0, 2, 3, 1).contiguous()

                loc.append(loc_pred_prev)
                conf.append(conf_pred_prev)
            #    time_list.append(time.time())
            #print('time consumption:{}'.format(time_list[stage+1]-time_list[stage]))
            stage+=1
        #print(time_list) [1566464858.9122813, 1566464859.016053, 1566464859.5947318, 1566464867.2177444, 1566464963.6269689]
        #print('Prediction with Feature Enhancement Done..!')
        #x[5]=[1,38,38,512] --> loc[0]=[1,38,38,16], conf[0]=[1,38,38,84]  # 4 for anchor
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds (8732*p map for each prediction)
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
           output = (
                loc.view(loc.size(0), -1, 4), #[1, 8732, 4]
                conf.view(conf.size(0), -1, self.num_classes), #[1, 8732,  21]
                self.priors
            )

        #self.targets=targets

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    #if x:[BATCH,1024,19,19], x_prev:[BATCH,512,38,38], lx_prev:[BATCH,38,38,4*4], cx_prev:[BATCH,38,38,4*21]
    #enhanceing feature x with relation feature of prev_x



    #input: origin_sources, origin_pred
    #return: enhanced_sources (with concat)
class get_enhanced_source(nn.Module):
    def __init__(self):
        super(get_enhanced_source, self).__init__()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,priors, targets, num_classes, source,cls_pred, loc_pred, state,use_gpu=True,
                  threshold=0.1, variance=[0.1,0.2]): #state from 0 to 5):
        _scale=[1*1,3*3,5*5,10*10,19*19,38*38]
        _prior_dim=[4,4,6,6,4]
        step = 0
        for i in range(state):
            step += _scale[i]*_prior_dim[i]
        split_loc=priors.size()[0] - step
        #separated_prior = priors[step:step + _scale[state]*_prior_dim[state], :]
        separated_prior = priors[split_loc:split_loc + _scale[state-1]*_prior_dim[state-1],:]
        #From now, 8732 is converted to specific level size
         #   num = loc_data.size(0) #batch_size
        num=loc_pred.size(0)
        #priors = separated_prior[:loc_pred.size(1), :]
        num_priors = (separated_prior.size(0)) #8732
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4) #[32,target_prior,4]
        conf_t = torch.LongTensor(num, num_priors) #[32,target_prior]


        for idx in range(num):
            truths = targets[idx][:, :-1].data #only for coordinates [3,4]
            labels = targets[idx][:, -1].data #only for class_idx # 3(#obj in a img)
            defaults = separated_prior.data
            match(threshold, truths, defaults, variance, labels,
                  loc_t, conf_t, idx)

            #conf_t -->to be used for labeling certain anchor region
        #if use_gpu:
        #    loc_t = loc_t.cuda()
        #    conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False) #loc_t: [BATCHSIZE, targetanchor, 4] target coordinate per target anchor
        conf_t = Variable(conf_t, requires_grad=False) #conf_t: [BATCHSIZE, targetanchor] target class idx per target anchor

        pos = conf_t > 0 # 0 for background
        region_pos_=pos.view(pos.size(0),-1,_scale[state-1])
        region_pos_sum=region_pos_.sum(dim=1)
        region_pos_final= region_pos_sum>0 # converted from anchor-wise f/g to region-wise f/g
        entire_conf_scores= self.softmax(cls_pred.view(cls_pred.size(0),-1,num_classes))
        regional_conf_scores=entire_conf_scores.view(entire_conf_scores.size()[0],cls_pred.size()[1]*cls_pred.size()[2],-1)
        region_max_score, region_max_cls = regional_conf_scores.max(dim=-1)
        region_max_cls= region_max_cls%(num_classes)
        #print("region_max_cls per state:{}".format(region_max_cls))
        local_graph=build_local_graph(region_max_cls) #time consumption toooooooooo large

        source=source.permute(0, 2, 3, 1).contiguous()
        source=source.view(source.size()[0],source.size()[1]*source.size()[2],-1)
        fused_buffer = torch.bmm(local_graph, source)
        #time_list.append(time.time())
        #time_per_line=list()
        #for q in range(len(time_list)-1):
        #    time_per_line.append(time_list[q+1]-time_list[q])
        #plt.plot(time_per_line)
        #plt.show()
        return fused_buffer, source




# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    loc_layers.reverse()
    conf_layers.reverse()
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_relation_SSD(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    scale = [256,256,256,512,1024,512]
    decon_kernel = [3,3,2,3,2]
    decon_pad=[0,1,0,1,0]
    rel_ = []
    upscalingConv_ = []
    bufConv_ = []
    for i in range(5):
        rel_ += [nn.Linear(scale[i],int(scale[i]/8))]
        upscalingConv_ += [nn.ConvTranspose2d(scale[i]+int(scale[i]/8),scale[i+1], stride=2,
                               kernel_size= decon_kernel[i],padding=decon_pad[i])]
        bufConv_ += [nn.Conv2d(scale[i+1],scale[i+1], (3,3), padding=1)]
    return relation_SSD(phase, size, base_, extras_, head_, num_classes,
                        rel_, upscalingConv_, bufConv_)


def build_local_graph2(class_idx): #ex: [64,25]
    graph=torch.zeros((class_idx).size()[0],class_idx.size()[1], class_idx.size()[1])
    for n in range(class_idx.size()[0]):
        for i in range(class_idx.size()[1]):
            for j in range(class_idx.size()[1]):
                graph[n][i][j]=cls_r_prob[class_idx[n][i]][class_idx[n][j]]
    return graph

#Fast Mode but, [bg,bg]->1 as well
def build_local_graph(class_idx):  # ex: [64,25]
    batch_size = class_idx.size()[0]
    graph_len = class_idx.size()[1]
    graph = torch.zeros(batch_size, graph_len, graph_len)
    for n in range(batch_size):
        identity_m = torch.eye(graph_len, graph_len)
        for i in range(graph_len):
            for j in range(i):
                graph[n][i][j] = cls_r_prob[class_idx[n][i]][class_idx[n][j]]
            if (class_idx[n][i]==0):
                identity_m[i][i]=0
        graph[n]=graph[n]+graph[n].permute(1,0)
        graph[n]+=identity_m

    return graph