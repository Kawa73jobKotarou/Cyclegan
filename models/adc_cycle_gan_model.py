import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class ADCCycleGANModel(BaseModel):
    """
    ADC-CycleGAN Model class.
    
    This model extends the standard CycleGAN to synthesize ADC images from MRI (and vice versa).
    The core architectural improvement (CBAM) is handled within the generator definitions 
    in 'networks.py'. This class manages the training loop, losses, and optimizers.
    
    Original Repo: https://github.com/JiayuanWang-JW/ADC-cycleGAN
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        ADC-CycleGAN 用のコマンドラインオプション設定
        """
        parser.set_defaults(no_dropout=True)
        
        if is_train:
            # Cycle Consistency Loss の重み (A->B->A, B->A->B)
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            
            # デュアルコントラスト損失 (DC Loss) の重み [cite: 212]
            parser.add_argument('--lambda_DC', type=float, default=0.5, help='weight for Dual Contrast Loss') 
            
            # Identity Loss (論文にはないが残す)
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping.')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # ログに出力する損失関数の名前リスト
        # D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B はそのまま
        # DC_AとDC_Bを追加 (D_AはDC_B, D_BはDC_Aからの学習に寄与するため、Dの損失としてDCを分離)
        # 論文ではDCはDの更新のみに影響を与える
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'DC_A', 'DC_B']
        
        # 可視化する画像の名前リスト (HTMLやVisdom用)
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        
        # Identity Lossを使用する場合は、アイデンティティマッピングの結果も可視化します
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B') # G_A(B) -> B (Bを入れたらBが出るべき)
            visual_names_B.append('idt_A') # G_B(A) -> A (Aを入れたらAが出るべき)

        self.visual_names = visual_names_A + visual_names_B
        
        # 保存・読み込みを行うネットワークのリスト
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # テスト時はGeneratorのみロード
            self.model_names = ['G_A', 'G_B']

        # -------------------------------------------------------------------
        # ネットワークの定義 (Generator & Discriminator)
        # -------------------------------------------------------------------
        # ここで networks.define_G を呼び出す際、前回の変更により
        # 'resnet_9blocks' を指定すると CBAM 付きの ResNet が生成されます。
        # Generator A: MRI -> ADC
        self.netG_A = networks.define_G(opt,opt.input_nc, opt.output_nc, opt.ngf, 'resnet_9blocks_cbam', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # Generator B: ADC -> MRI
        self.netG_B = networks.define_G(opt,opt.output_nc, opt.input_nc, opt.ngf, 'resnet_9blocks_cbam', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Discriminator A & B (PatchGAN)
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # -------------------------------------------------------------------
        # 損失関数とオプティマイザの定義
        # -------------------------------------------------------------------
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
                
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # Loss Functions
            # GAN Loss は標準を使用 (Dの更新にDCを組み込む)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device) 
            
            # Cycle Consistency Loss は SSIMベースの損失 (1 - SSIM) に変更が必要 [cite: 335]
            # ここでは便宜上、標準L1を残し、SSIMの実装が必要であることを示します
            # L_cycle(G,F) = (1 - SSIM(F(G(x)), x)) + (1 - SSIM(G(F(y)), y)) [cite: 335]
            self.criterionCycle = networks.SSIMLoss() # SIM Loss (1 - SSIM) に置き換えるべき [cite: 335]
            self.criterionIdt = torch.nn.L1Loss()   # Identity Loss (L1)
            
            # Optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Dataloaderからの入力をセット"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Forward pass: 生成画像の計算"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A) -> 生成ADC
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A)) -> 再構成MRI
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) -> 生成MRI
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B)) -> 再構成ADC

    def backward_D_basic(self, netD, real_target, fake_target, real_source):
        """Discriminatorの損失計算ヘルパー（ADC-CycleGAN版）
        
        Parameters:
            netD (nn.Module): 判別器 (D_A または D_B)
            real_target (Tensor): ターゲットドメインの実画像 (y または x)
            fake_target (Tensor): ターゲットドメインの合成画像 (ŷ または x̂)
            real_source (Tensor): ソースドメインの実画像 (x' または y')
            
        """
        lambda_DC = self.opt.lambda_DC
        
        # 1. GAN Loss の計算 (リアル vs. フェイク)
        # Real (本物) - クラス 1
        pred_real = netD(real_target)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake (偽物 - 合成画像) - クラス 0
        pred_fake = netD(fake_target.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # 2. Dual Contrast Loss (DC Loss) の計算
        # Source Domain Real Image (ソースドメインの実画像) - クラス 0 [cite: 205]
        # netD(real_source) がクラス 0 と判定されるように学習 
        pred_source = netD(real_source.detach())
        loss_DC = self.criterionGAN(pred_source, False)
        
        # 3. 統合
        # D_Loss = (GAN_Real + GAN_Fake) * 0.5 + lambda_DC * DC_Loss
        loss_D = (loss_D_real + loss_D_fake) * (1-lambda_DC) + loss_DC * lambda_DC
        
        loss_D.backward()
        
        # DC Lossの値をログ用に別に保存（D_Lossには含まれているが、個別の値も必要）
        return loss_D, loss_DC

    def backward_D_A(self):
        """D_A (Bドメイン/ADC画像の真贋判定) のLoss計算"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        # D_Aはターゲット B (real_B, fake_B) と、ソース A (real_A) から学習
        self.loss_D_A, self.loss_DC_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.real_A)
        # ログ用に loss_DC_A を更新

    def backward_D_B(self):
        """D_B (Aドメイン/MRI画像の真贋判定) のLoss計算"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        # D_Bはターゲット A (real_A, fake_A) と、ソース B (real_B) から学習
        self.loss_D_B, self.loss_DC_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.real_B)
        # ログ用に loss_DC_B を更新

    def backward_G(self):
        """Generator G_A と G_B のLoss計算"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # 1. Identity Loss (構造維持)
        # G_Aに本物のB(ADC)を入れたら、何も変えずにBを出してほしい
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # 2. GAN Loss (Dを騙す)
        # D_A(G_A(A)) -> Trueと判定させたい
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # D_B(G_B(B)) -> Trueと判定させたい
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # || G_B(G_A(A)) - A || (SSIMベース)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # || G_A(G_B(B)) - B || (SSIMベース)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Total Loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """パラメータ更新の1ステップ"""
        # Forward pass
        self.forward()

        # Generatorsの更新
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Dの勾配計算をオフにする
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Discriminatorsの更新
        self.set_requires_grad([self.netD_A, self.netD_B], True)   # Dの勾配計算をオンにする
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()