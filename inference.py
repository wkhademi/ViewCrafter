from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    
    avg_psnr = []
    avg_ssim = []
    avg_lpips = []
    for scene_id in ["1ada7a0617", "3f15a9266d", "5f99900f09", "45b0dac5e3", "6115eddb86"]:
        opts.save_dir = os.path.join(opts.out_dir, opts.exp_name, scene_id)
        os.makedirs(opts.save_dir,exist_ok=True)
        opts.scene_id = scene_id
        pvd = ViewCrafter(opts)
        _, eval_metrics =pvd.nvs_sparse_view_eval()
        avg_psnr.append(eval_metrics['psnr'])
        avg_ssim.append(eval_metrics['ssim'])
        avg_lpips.append(eval_metrics['lpips'])
        del pvd
        torch.cuda.empty_cache()
    print(f'Average PSNR: {np.mean(avg_psnr):.3f} dB')
    print(f'Average SSIM: {np.mean(avg_ssim):.4f}')
    print(f'Average LPIPS: {np.mean(avg_lpips):.4f}')
