import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleLSeSimModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            #VGG16のパラメータ
            parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
            parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
            parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
            parser.add_argument('--model_vgg16', type=str, default="image_net", help='|image_net| or model file name')
            parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
            parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
            parser.add_argument('--learned_attn', action='store_true', help='use the learnable attention map')
            parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
            parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
            parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for feature consistency loss')
            parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss')
            parser.add_argument('--use_CTloss', action='store_true', help='use the CT loss')
            parser.add_argument('--lambda_CT', type=float, default=10.0, help='weight for CT loss')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'style', 'G_A_s', 'per']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.isTrain and self.opt.use_CTloss:
            self.loss_names.append('CT')

        self.input_nc = opt.input_nc


        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
            self.netPre = networks.VGG16(opt.model_vgg16).to(self.device)
            

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # LSeSimで追加されたネットワーク
            self.criterionStyle = networks.StyleLoss(opt.model_vgg16).to(self.device)
            self.criterionFeature = networks.PerceptualLoss(opt.model_vgg16).to(self.device)
            self.criterionSpatial = networks.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = networks.Normalization(self.device)
            # realCTとfakeCTでpixelベースのlossを取得
            self.criterionCT = torch.nn.L1Loss()

            # define the contrastive loss
            if opt.learned_attn:
                self.netF = self.criterionSpatial
                self.model_names.append('F')
                self.loss_names.append('spatial')
            else:
                self.set_requires_grad([self.netPre], False)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    # Generator や attention ネットワーク（netF, netPre）の学習準備.特徴マップのサイズに依存した重みやバッファを初期化するために、初回の画像入力とフォワード処理を行う。
    def data_dependent_initialize(self, data):
            """
            The learnable spatially-correlative map is defined in terms of the shape of the intermediate, extracted features
            of a given network (encoder or pretrained VGG16). Because of this, the weights of spatial are initialized at the
            first feedforward pass with some input images
            :return:
            """
            self.set_input(data)
            bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
            self.real_A = self.real_A[:bs_per_gpu]
            self.real_B = self.real_B[:bs_per_gpu]
            self.forward()
            if self.isTrain:
                self.backward_G()
                self.optimizer_G.zero_grad()
                if self.opt.learned_attn:
                    self.optimizer_F = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad, self.netPre.parameters())), 'lr': self.opt.lr*0.0},
                                            {'params': list(filter(lambda p:p.requires_grad, self.netF.parameters()))}],
                                            lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                    self.optimizers.append(self.optimizer_F)
                    self.optimizer_F.zero_grad()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_F(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        if self.input_nc == 1:
            norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B.detach() + 1) * 0.5)
        else:
            center_channel = (self.input_nc-1)/2
            center_channel = int(center_channel)
            norm_real_A = self.normalization((self.real_A[:, center_channel:center_channel+1, :, :] + 1)* 0.5)
            norm_real_B, norm_fake_B = self.normalization((self.real_B[:, center_channel:center_channel+1, :, :] + 1) * 0.5), self.normalization((self.fake_B[:, center_channel:center_channel+1, :, :].detach() + 1) * 0.5)

        self.loss_spatial = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, norm_real_B)

        self.loss_spatial.backward()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # LSeSimで追加
        l_style = self.opt.lambda_style
        l_per = self.opt.lambda_perceptual
        l_sptial = self.opt.lambda_spatial
        # CTのpixelベースの損失
        l_CT = self.opt.lambda_CT

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # LSeSimで追加（style loss, perceptial loss, structure loss）
        if self.input_nc == 1:
            norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        else:
            center_channel = (self.input_nc-1)/2
            center_channel = int(center_channel)
            norm_real_A = self.normalization((self.real_A[:, center_channel:center_channel+1, :, :] + 1)* 0.5)
        norm_fake_B = self.normalization((self.fake_B[:, center_channel:center_channel+1, :, :] + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B[:, center_channel:center_channel+1, :, :] + 1) * 0.5)
        self.loss_style = self.criterionStyle(norm_real_B, norm_fake_B) * l_style if l_style > 0 else 0
        self.loss_per = self.criterionFeature(norm_real_A, norm_fake_B) * l_per if l_per > 0 else 0
        self.loss_G_A_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * l_sptial if l_sptial > 0 else 0
        # CTのpixelベースの損失
        self.loss_CT = self.criterionCT(self.fake_B,self.real_B) * l_CT if self.opt.use_CTloss else 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_per + self.loss_style + self.loss_G_A_s + self.loss_CT
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # self.opt.learned_attn = Falseでした
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], True)
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], False)
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers