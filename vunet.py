import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.random import normal
from math import sqrt
import argparse

#channel_dim=3
#ndf=32

class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        #combine two paths
        x = x_l + x_r
        return x

###################################################################################################################
class unet_conv_block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation, prob_dropout, prob_leaky, sign_normalization):
        super(unet_conv_block, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2

        self.prob_dropout = prob_dropout
        self.prob_leaky = prob_leaky
        self.sign_normalization = sign_normalization

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=(pad0, pad1), dilation=dilation)
        self.dropout = nn.Dropout(p=prob_dropout)

        if self.prob_leaky < 0.000001:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(prob_leaky, inplace=True)

        if self.sign_normalization == 1:
            self.normalization=nn.BatchNorm2d(out_dim)

        if self.sign_normalization == 2:
            self.normalization=nn.InstanceNorm2d(out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        if self.sign_normalization >0 :
           x = self.normalization(x)        

        x = self.relu(x)
        return x

#########################################################################################################
class unet_512(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf, prob_dropout, sign_normalization, sign_upsample):
        super(unet_512, self).__init__()
        self.ngpu = ngpu
        self.channel_dim = channel_dim
        self.ndf = ndf
        self.prob_dropout = prob_dropout
        self.sign_normalization = sign_normalization #0 for none, 1 for batch normalization 2 for instance normalization
        self.sign_upsample = sign_upsample #0 for upsample 1 for transpose convolution
        self.num_layers = 4
        self.kernel_size = [3, 3]
        self.dilation = 1
        self.prob_leaky=0.02

        self.encoder_layer1 = nn.Sequential(
            unet_conv_block(self.channel_dim, self.ndf, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf,         self.ndf, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
        ### ndf*512*512

        self.encoder_layer1_downsample = nn.MaxPool2d(2, stride=2)
        # ndfx256x256

        self.encoder_layer2 = nn.Sequential(
            unet_conv_block(self.ndf,   self.ndf*2, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*2, self.ndf*2, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
	#2ndf*256*256
        
        self.encoder_layer2_downsample = nn.MaxPool2d(2, stride=2)
        # 2ndfx128x128
	
        self.encoder_layer3 = nn.Sequential(
            unet_conv_block(self.ndf*2,  self.ndf*4, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*4,  self.ndf*4, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
        #4ndf*128*128

        self.encoder_layer3_downsample = nn.MaxPool2d(2, stride=2)
        # 4ndfx64x64

        self.encoder_layer4 = nn.Sequential(
            unet_conv_block(self.ndf*4,  self.ndf*8, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*8,  self.ndf*8, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
        # 8ndf*64*64

        self.encoder_layer4_downsample = nn.MaxPool2d(2, stride=2)
        # 8ndf*32*32
		
        self.encoder_layer5 = nn.Sequential(
            unet_conv_block(self.ndf*8,  self.ndf*16, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*16, self.ndf*16, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
        # 16ndf*32*32

        self.encoder_layer5_downsample = nn.MaxPool2d(2, stride=2)
        # 16ndf*16*16

        self.encoder_lastlayer = nn.Sequential(        
            unet_conv_block(self.ndf*16,  self.ndf*32, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*32, self.ndf*32, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )
        # 16ndf*16*16

        ##################################################################################################
        pad0 = (self.kernel_size[0] - 1) // 2
        pad1 = (self.kernel_size[1] - 1) // 2

        self.decoder_layer5_conv1 = nn.Conv2d(self.ndf*32, self.ndf*16, self.kernel_size, padding=(pad0, pad1), dilation=self.dilation)

        self.decoder_layer5 = nn.Sequential(
            unet_conv_block(self.ndf*32,  self.ndf*16, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*16,   self.ndf*16, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )

        self.decoder_layer4_conv1 = nn.Conv2d(self.ndf*16, self.ndf*8, self.kernel_size, padding=(pad0, pad1), dilation=self.dilation)

        self.decoder_layer4 = nn.Sequential(
            unet_conv_block(self.ndf*16,  self.ndf*8, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*8,   self.ndf*8, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )

        self.decoder_layer3_conv1 = nn.Conv2d(self.ndf*8, self.ndf*4, self.kernel_size, padding=(pad0, pad1), dilation=self.dilation)

        self.decoder_layer3 = nn.Sequential(
            unet_conv_block(self.ndf*8,  self.ndf*4, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*4,  self.ndf*4, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )

        self.decoder_layer2_conv1 = nn.Conv2d(self.ndf*4, self.ndf*2, self.kernel_size, padding=(pad0, pad1), dilation=self.dilation)
        
        self.decoder_layer2 = nn.Sequential(
            unet_conv_block(self.ndf*4,  self.ndf*2, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*2,  self.ndf*2, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )

        self.decoder_layer1_conv1 = nn.Conv2d(self.ndf*2, self.ndf, self.kernel_size, padding=(pad0, pad1), dilation=self.dilation)

        self.decoder_layer1 = nn.Sequential(
            unet_conv_block(self.ndf*2,  self.ndf*1, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization),
            unet_conv_block(self.ndf*1,  self.ndf*1, self.kernel_size, self.dilation, self.prob_dropout, self.prob_leaky, self.sign_normalization)
        )

        self.conv_final = nn.Conv2d(ndf, 1,  kernel_size=1, padding=0, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) | isinstance(m, nn.InstanceNorm2d) :
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
   
    def forward(self, input):

            encoder_layer1_output = self.encoder_layer1(input)        
            #print("encoder_layer1_output size ", encoder_layer1_output.size())

            encoder_layer2_input = self.encoder_layer1_downsample(encoder_layer1_output)
            #print("encoder_layer2_input size ", encoder_layer2_input.size())

            encoder_layer2_output = self.encoder_layer2(encoder_layer2_input)
            #print("encoder_layer2_output size ", encoder_layer2_output.size())

            encoder_layer3_input = self.encoder_layer2_downsample(encoder_layer2_output)
            #print("encoder_layer3_input size ", encoder_layer3_input.size())

            encoder_layer3_output = self.encoder_layer3(encoder_layer3_input)
            #print("encoder_layer3_output size ", encoder_layer3_output.size())
	    
            encoder_layer4_input = self.encoder_layer3_downsample(encoder_layer3_output)
            #print("encoder_layer4_input size ", encoder_layer4_input.size())

            encoder_layer4_output = self.encoder_layer4(encoder_layer4_input)
            #print("encoder_layer4_output size ", encoder_layer4_output.size())

            encoder_layer5_input = self.encoder_layer4_downsample(encoder_layer4_output)
            #print("encoder_layer5_input size ", encoder_layer5_input.size())

            encoder_layer5_output = self.encoder_layer5(encoder_layer5_input)
            #print("encoder_layer5_output size ", encoder_layer5_output.size())

            #######################################################
            encoder_lastlayer_input = self.encoder_layer5_downsample(encoder_layer5_output)
            #print("encoder_lastlayer_input size ", encoder_lastlayer_input.size())    

            encoder_lastlayer_output = self.encoder_lastlayer(encoder_lastlayer_input)	            
            #print("encoder_lastlayer_output size ", encoder_lastlayer_output.size())    

            #######################################################
            decoder_layer5_input = F.interpolate(encoder_lastlayer_output, size=encoder_layer5_output.size()[2:], mode='bilinear')
            #print("1 decoder_layer5_input size ", decoder_layer5_input.size())    
 
            decoder_layer5_input = self.decoder_layer5_conv1(decoder_layer5_input) 
            #print("2 decoder_layer5_input size ", decoder_layer5_input.size())    
            decoder_layer5_input = torch.cat([decoder_layer5_input, encoder_layer5_output], 1)
            #print("3 decoder_layer5_input size ", decoder_layer5_input.size())    
            decoder_layer5_output = self.decoder_layer5(decoder_layer5_input);
            #print("4 decoder_layer5_output size ", decoder_layer5_output.size())    

            #######################################################
            decoder_layer4_input = F.interpolate(decoder_layer5_output, size=encoder_layer4_output.size()[2:], mode='bilinear')
            decoder_layer4_input = self.decoder_layer4_conv1(decoder_layer4_input) 
            decoder_layer4_input = torch.cat([decoder_layer4_input, encoder_layer4_output], 1)
            decoder_layer4_output = self.decoder_layer4(decoder_layer4_input);

            #######################################################
            decoder_layer3_input = F.interpolate(decoder_layer4_output, size=encoder_layer3_output.size()[2:], mode='bilinear')
            decoder_layer3_input = self.decoder_layer3_conv1(decoder_layer3_input)
            decoder_layer3_input = torch.cat([decoder_layer3_input, encoder_layer3_output], 1)
            decoder_layer3_output = self.decoder_layer3(decoder_layer3_input);	    

            #######################################################
            decoder_layer2_input = F.interpolate(decoder_layer3_output, size=encoder_layer2_output.size()[2:], mode='bilinear')
            decoder_layer2_input = self.decoder_layer2_conv1(decoder_layer2_input)
            decoder_layer2_input = torch.cat([decoder_layer2_input, encoder_layer2_output], 1)	               
            decoder_layer2_output = self.decoder_layer2(decoder_layer2_input)

            #######################################################
            decoder_layer1_input = F.interpolate(decoder_layer2_output, size=encoder_layer1_output.size()[2:], mode='bilinear')
            decoder_layer1_input = self.decoder_layer1_conv1(decoder_layer1_input)
            decoder_layer1_input = torch.cat([decoder_layer1_input, encoder_layer1_output], 1)	               
            decoder_layer1_output = self.decoder_layer1(decoder_layer1_input)

            #######################################################
            output = F.sigmoid(self.conv_final(decoder_layer1_output))

            return output

