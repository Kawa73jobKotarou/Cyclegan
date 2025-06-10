import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import re
import pdb


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

# def save_dicom_images(image_dir, visuals, image_path, opt, B_path):
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#     label = "pseudo-CT"
#     os.makedirs(os.path.join(image_dir, label), exist_ok=True)
#     dicom_name = '%s/%s.dcm' % (label, name)
#     dicom_save_path = os.path.join(image_dir, dicom_name)
#     im_data = visuals['fake_B']
#     util.save_dicom_image(im_data, dicom_save_path, opt, B_path)

def save_dicom_images(image_dir, visuals, image_path, opt):
    if opt.input_nc > 1:
        image_path = image_path[int((opt.input_nc-1)/2)]
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    label = "pseudo-CT"
    os.makedirs(os.path.join(image_dir, label), exist_ok=True)
    match = re.search(r'(\d+_\d+)', image_path[0])
    if match:
        result = match.group(1)
    # dicom_name = '%s/%s/%s.dcm' % (label, result, name)
    image_dir=os.path.join(image_dir, label,result)
    os.makedirs(image_dir, exist_ok=True)
    dicom_name = '%s.dcm' % (name)
    dicom_save_path = os.path.join(image_dir, dicom_name)
    im_data = visuals['fake_B']
    B_path = image_path[0]
    B_path = B_path.replace("testA", "testB")
    B_path = B_path.replace("MR", "CT")
    if "CTCAT" in B_path:
        B_path = B_path.replace("CTCAT", "MRCAT")
    util.save_dicom_image(im_data, dicom_save_path, opt, B_path)

def save_nifti_images(image_dir, visuals, image_path, opt, affine):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    label = "pseudo-CT"
    os.makedirs(os.path.join(image_dir, label), exist_ok=True)
    nifti_name = '%s/%s.gz' % (label, name)
    nifti_save_path = os.path.join(image_dir, nifti_name)
    im_data = visuals['fake_B']
    util.save_nifti_image(im_data, nifti_save_path, opt, affine)


def save_eps_images(image_dir, visuals, image_path, opt):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    name = os.path.splitext(name)[0]
    label = "eps"
    os.makedirs(os.path.join(image_dir, label), exist_ok=True)
    eps_name = '%s/%s.eps' % (label, name)
    eps_save_path = os.path.join(image_dir, eps_name)
    im_data = visuals['fake_B']
    util.save_eps_image(im_data, eps_save_path, opt)

def save_images(webpage, visuals, image_path, opt, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    image_path_0 = os.path.abspath(image_path[0])
    # "testA" の位置を探す
    split_token = os.sep + 'testA' + os.sep
    if split_token in image_path_0:
        relative_path = image_path_0.split(split_token)[1]  # → '03_01/MR_03_01.dcm'
    else:
        raise ValueError(f"'testA' not found in path: {image_path_0}")
    name = os.path.splitext(relative_path)[0]  # → 03_01/MR_03_01
    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        im = im[:, :, int((opt.input_nc - 1) / 2)]
        # 拡張子付きファイル名を作成
        image_name = f"{name}_{label}.png"  # → 03_01/MR_03_01_real_A.png
        # サブフォルダ付きで保存パスを作成
        save_path = os.path.join(image_dir, image_name)
        # サブフォルダがなければ作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.final_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'final_loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, mr_bone_flag = 0, ct_bone_flag = 0):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    if self.opt.input_nc != 1:
                        center = (self.opt.input_nc-1)/2
                        center = int(center)
                        center_images = images[0][center]
                        center_images = np.stack([center_images, center_images, center_images], axis=0)
                        images[0]=center_images
                        
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                # if self.opt.input_nc != 1 and label == "real_A":
                if self.opt.input_nc != 1:
                    center = (self.opt.input_nc-1)/2
                    center = int(center)
                    image = image[:,center:center+1,:,:]
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    # if self.opt.input_nc != 1 and label == "real_A":
                    if self.opt.input_nc != 1:
                        center = (self.opt.input_nc-1)/2
                        center = int(center)
                        image_numpy = image_numpy[:,center:center+1,:,:]
                    image_numpy = util.tensor2im(image_numpy)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    if label == "real_A":
                        txts.append(label + " " +str(mr_bone_flag))
                    elif label == "real_B":
                        txts.append(label + " " +str(ct_bone_flag))
                    else:
                        txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_save_epoch(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        # ロスが出力されているところ
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.7f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    # 新しく追加
    # 各epochの最後のロスを保存
    def save_final_losses(self, epoch, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        # ロスが出力されているところ
        message = '%d ' % (epoch)
        for k, v in losses.items():
            message += '%.7f ' % (v)

        with open(self.final_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
