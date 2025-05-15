import sys

# Maybe use flowpy instead which can read/write png and flo files in KITTI and Middleburry format respectively.
sys.path.append('/workspace/scenes')

# append path to current directory
sys.path.append('evaluate')

import csv
import yaml
import flow_vis  # flow_vis library implements Middlebury flow vector coloring
from tfds_util import *
from error_measures import *
import tensorflow_datasets as tfds


class FigureConfig:
    def __init__(self, video_name, video_type, n_rows, n_cols):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.name = video_name
        self.dir = video_type
        self.set_size(n_rows, n_cols)
        self.img_cnt = 0

    def set_size(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig_id, self.fig_axs = plt.subplots(self.n_rows, self.n_cols, figsize=(self.n_cols, self.n_rows))
        self.fig_id.subplots_adjust(wspace=0, hspace=0)

    def get_axis(self, index: int):
        # index starts at 1
        row = int((index - 1) / self.n_cols)
        col = index - row * self.n_cols - 1
        return self.fig_axs[row, col]

    @classmethod
    def get_flow_fig(cls, cfg, t_flow):
        n_cols = 4
        if cfg.figure.quiver:
            n_cols += 2
            res = cls(t_flow.video_name + '_flow', t_flow.video_type, t_flow.num_frames - 1, n_cols)
            res.x_mesh = np.empty(1)
            res.y_mesh = np.empty(1)
        else:
            res = cls(t_flow.video_name + '_flow', t_flow.video_type, t_flow.num_frames - 1, n_cols)
        return res

    @classmethod
    def get_epe_fig(cls, cfg, t_flow):
        # n_cols depends on the options selected
        n_cols = 1 + int(cfg.figure.fg) + int(cfg.figure.bg)
        n_cols *= (1 + int(cfg.figure.relative))
        res = cls(t_flow.video_name + '_epe', t_flow.video_type, t_flow.num_frames - 1, n_cols)
        return res

    @classmethod
    def get_sintel_fig_debug(cls, cfg, t_flow):
        # n_cols is fixed
        n_cols = 6
        res = cls(t_flow.video_name + '_sintel', t_flow.video_type, t_flow.num_frames - 1, n_cols)
        return res

    @classmethod
    def get_sintel_fig(cls, cfg, t_flow):
        # n_cols is fixed
        n_cols = 8
        res = cls(t_flow.video_name + '_sintel', t_flow.video_type, t_flow.num_frames - 1, n_cols)
        return res


class TestFlow:
    SMALL_FLOW = 5.0
    MEDIUM_FLOW = 15.0
    CLOSE_BND = 5.0
    MEDIUM_BND = 30.0
    FAR_BND = 70.0
    STR_SMALL_FLOW = "s0_{:.0f}".format(SMALL_FLOW)
    STR_MEDIUM_FLOW = "s{:.0f}_{:.0f}".format(SMALL_FLOW, MEDIUM_FLOW)
    STR_LARGE_FLOW = "s{:.0f}_".format(MEDIUM_FLOW)
    STR_CLOSE_BND = "d0_{:.0f}".format(CLOSE_BND)
    STR_MEDIUM_BND = "d{:.0f}_{:.0f}".format(CLOSE_BND, MEDIUM_BND)
    STR_FAR_BND = "d{:.0f}_{:.0f}".format(MEDIUM_BND, FAR_BND)

    def __init__(self, cfg):
        # Name needs forward slash independent of OS
        cwd_path = os.getcwd()
        print("[test_flow] Runtime directory: ", cwd_path)

        # for path_entry in os.listdir(DATASET_BASE_PATH):
        #     print("[test_flow]", path_entry)

        full_path = cfg.base + '/' + str(cfg['tfds']) + '/' + str(cfg['tfds_config'])
        print("[test_flow] Full dataset path:", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError("The specified TFDS not found.")

        self.flow_dataset = tfds.load(cfg['tfds'] + '/' + str(cfg['tfds_config']))
        # self.flow_dataset = load_all_datasets(base_path)
        self.cfg = cfg
        if cfg['predictor'] == 'arflow':
            import predict_arflow
            self.predictor = predict_arflow.PredictARFlow()
        elif cfg['predictor'] == 'gmflow':
            import predict_gmflow
            self.predictor = predict_gmflow.PredictGMFlow()
        elif cfg['predictor'] == 'raft':
            import predict_raft
            self.predictor = predict_raft.PredictRAFT()
        elif cfg['predictor'] == 'gma':
            import predict_gma
            self.predictor = predict_gma.PredictGMA()

        self.csvfile = None
        self.writer = None
        self.row_dict = None

        try:
            self.with_figures = cfg.figure.images or cfg.figure.quiver or cfg.figure.epe or cfg.figure.fg or cfg.figure.bg or cfg.figure.relative or cfg.figure.sintel or cfg.figure.sintel_debug
        except:
            self.with_figures = False
            # self.with_sintel = cfg['defaults'].error.mpi_sintel == "jacobian" or cfg['defaults'].error.mpi_sintel == "divergence"

    def __del__(self):
        self.csvfile.close()
        return

    def open_csv(self):
        csv_path = "/workspace/flow_evals"
        titles = ["video_type", "video_name", "epe", "epe_rel", "epe_fg",
                  "epe_fg_rel", "epe_bg", "epe_bg_rel", "matched", "unmatched",
                  self.STR_CLOSE_BND, self.STR_MEDIUM_BND, self.STR_FAR_BND,
                  self.STR_SMALL_FLOW, self.STR_MEDIUM_FLOW, self.STR_LARGE_FLOW]

        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        self.csvfile = open(os.path.join(
            csv_path, self.cfg['tfds_config'] + "_" + self.cfg['predictor'] + ".csv"
        ), 'w', newline='')

        self.writer = csv.DictWriter(self.csvfile, fieldnames=titles)
        self.writer.writeheader()
        self.row_dict = dict.fromkeys(titles, 0)

    def write_stats(self):
        if self.num_frames > 0:
            for k, v in self.row_dict.items():
                if k == "video_type" or k == "video_name":
                    continue
                self.row_dict[k] = v / self.num_frames
            print("[test_flow] EPE:", self.row_dict['epe'])
        self.writer.writerow(self.row_dict)
        self.csvfile.flush()  # Ensure data is written to the file

    def setup_figures(self):
        if self.cfg.figure.images:
            self.fig_flow = FigureConfig.get_flow_fig(self.cfg, self)
        if self.cfg.figure.epe:
            self.fig_epe = FigureConfig.get_epe_fig(self.cfg, self)
        if self.cfg.figure.sintel and self.cfg.error.mpi_sintel != "none":
            self.fig_sintel = FigureConfig.get_sintel_fig(self.cfg, self)
        if self.cfg.figure.sintel_debug and self.cfg.error.mpi_sintel != "none":
            self.fig_sintel_debug = FigureConfig.get_sintel_fig_debug(self.cfg, self)
        return

    def save_figures(self):
        if self.cfg.figure.images:
            save_fig(self.fig_flow.fig_id, self.video_name + '_flow_' + self.cfg.predictor, self.video_type)
            plt.close(self.fig_flow.fig_id)  # close the figure to free up memory
        if self.cfg.figure.epe:
            save_fig(self.fig_epe.fig_id, self.video_name + '_epe_' + self.cfg.predictor, self.video_type)
            plt.close(self.fig_epe.fig_id)
        if self.cfg.figure.sintel:
            save_fig(self.fig_sintel.fig_id, self.video_name + '_sintel_' + self.cfg.predictor, self.video_type)
            plt.close(self.fig_sintel.fig_id)
        if self.cfg.figure.sintel_debug:
            save_fig(self.fig_sintel_debug.fig_id, self.video_name + '_sintel_debug_' + self.cfg.predictor,
                     self.video_type)
            plt.close(self.fig_sintel_debug.fig_id)
        return

    def calc_error(self, flow_pred, forward_flow, mask, fg_mask, bg_mask):
        flow_error, epe = calc_epe(flow_pred, forward_flow, mask)
        self.row_dict['epe'] += epe

        if self.with_figures and self.cfg.figure.epe:
            self.fig_epe.img_cnt += 1
            ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
            _ = ax.imshow(flow_error)
            ax.axis("equal")
            ax.axis("off")
            # ax.set_title("EPE: {:05.3f}".format(epe), fontsize=6)

        # Visualize flows
        # fid = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.title("Predicted Flow")
        # plt.imshow(flow_vis.flow_to_color(flow_pred, convert_to_bgr=False))
        # plt.subplot(1, 2, 2)
        # plt.title("Ground Truth Flow")
        # plt.imshow(flow_vis.flow_to_color(forward_flow, convert_to_bgr=False))
        # save_fig(fid, 'test')

        flow_error, epe = calc_epe(flow_pred, forward_flow, mask)
        self.row_dict['epe'] += epe

        # if False:  # self.cfg.error.relative :
        #     flow_error_rel, epe_rel = calc_rel_epe(
        #         flow_pred, forward_flow, flow_error, mask)
        #     self.row_dict['epe_rel'] += epe_rel
        #     if self.with_figures and self.cfg.figure.epe:
        #         self.fig_epe.img_cnt += 1
        #         ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
        #         _ = ax.imshow(flow_error_rel)
        #         ax.axis("equal")
        #         ax.axis("off")
        #         # ax.set_title("Relative: {:05.3f}".format(epe_rel), fontsize=6)
        #
        # if False:  # self.cfg.error.fg :
        #     flow_error_fg, epe_fg = calc_epe(flow_pred, forward_flow, fg_mask)
        #     self.row_dict['epe_fg'] += epe_fg
        #     if self.with_figures and self.cfg.figure.epe:
        #         self.fig_epe.img_cnt += 1
        #         ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
        #         _ = ax.imshow(flow_error_fg)
        #         ax.axis("equal")
        #         ax.axis("off")
        #     if True:  # self.cfg.error.relative :
        #         flow_error_fg_rel, epe_fg_rel = calc_rel_epe(flow_pred,
        #                                                      forward_flow,
        #                                                      flow_error_fg,
        #                                                      fg_mask)
        #         self.row_dict['epe_fg_rel'] += epe_fg_rel
        #         if self.with_figures and self.cfg.figure.epe:
        #             self.fig_epe.img_cnt += 1
        #             ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
        #             _ = ax.imshow(flow_error_fg_rel)
        #             ax.axis("equal")
        #             ax.axis("off")
        #
        # if False:  # self.cfg.error.bg :
        #     flow_error_bg, epe_bg = calc_epe(flow_pred, forward_flow, bg_mask)
        #     self.row_dict['epe_bg'] += epe_bg
        #     if self.with_figures and self.cfg.figure.epe:
        #         self.fig_epe.img_cnt += 1
        #         ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
        #         _ = ax.imshow(flow_error_bg)
        #         ax.axis("equal")
        #         ax.axis("off")
        #     if True:  # self.cfg.error.relative :
        #         flow_error_bg_rel, epe_bg_rel = calc_rel_epe(flow_pred,
        #                                                      forward_flow,
        #                                                      flow_error_bg,
        #                                                      bg_mask)
        #         self.row_dict['epe_bg_rel'] += epe_bg_rel
        #         if self.with_figures and self.cfg.figure.epe:
        #             self.fig_epe.img_cnt += 1
        #             ax = self.fig_epe.get_axis(self.fig_epe.img_cnt)
        #             _ = ax.imshow(flow_error_bg_rel)
        #             ax.axis("equal")
        #             ax.axis("off")
        return flow_error

    def find_motion_boundary(self,
                             forward_flow,
                             backward_flow,
                             segmentation,
                             depth,
                             mask):
        # find changes in flow field
        if 'jacobian' == "jacobian":  # self.cfg.error.mpi_sintel == "jacobian" :
            j_det = l2_jacobian(forward_flow)
            # abs_jacobian_determinant(forward_flow)
            j_det_mask = np.ma.masked_less(j_det, 1e-8)
            j_det_mask = j_det_mask.compressed()
            # if mask is empty quantile will fail
            if j_det_mask.size > 0:
                upper = np.quantile(j_det_mask, 0.75)
                iqr = upper - np.quantile(j_det_mask, 0.25)
                # print( "Jacobian Max: {:07.5f} IQR: {:07.5f} Limit: {:07.5f}".format(np.max(j_det), iqr, upper + 1.5 * iqr))
                motion_chg = (j_det > upper + 1.5 * iqr)
            else:
                motion_chg = np.full(j_det.shape, False)
            # j_det = (j_det > 0.1)  # Threshold
            # j_det = j_det * mask2D
            if self.with_figures and self.cfg.figure.sintel_debug:
                self.fig_sintel_debug.img_cnt += 1
                ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
                _ = ax.imshow(motion_chg)
                ax.axis("equal")
                ax.axis("off")
        elif self.cfg.error.mpi_sintel == "divergence":
            # Divergence
            div_f = abs_divergence(forward_flow)
            div_f_mask = np.ma.masked_less(div_f, 1e-8)
            div_f_mask = div_f_mask.compressed()
            if div_f_mask.size > 0:
                upper = np.quantile(div_f_mask, 0.75)
                iqr = upper - np.quantile(div_f_mask, 0.25)
                # print( "Divergence Max: {:07.5f} IQR: {:07.5f} Limit: {:07.5f}".format(np.max(div_f),
                # np.mean(div_f_mask), upper + 1.5 * iqr))
                # div_f = (div_f > 0.5)  # Threshold is 2 pixels per frame (ppf) in Butler et al. 2012
                motion_chg = (div_f > upper + 1.5 * iqr)
            else:
                motion_chg = np.full(div_f.shape, False)
            # div_f = div_f * mask2D
            if self.with_figures and self.cfg.figure.sintel_debug:
                self.fig_sintel_debug.img_cnt += 1
                ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
                _ = ax.imshow(motion_chg)
                ax.axis("equal")
                ax.axis("off")
        else:
            raise ValueError(self.cfg.error.mpi_sintel)
        # boundary of objects
        bnd = segmentation * mask
        bnd = np.reshape(bnd, (bnd.shape[0], bnd.shape[1]))
        bnd = mag_grad(bnd)
        if self.with_figures and self.cfg.figure.sintel_debug:
            self.fig_sintel_debug.img_cnt += 1
            ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
            _ = ax.imshow(bnd)
            ax.axis("equal")
            ax.axis("off")
        # depth boundaries
        depth = depth * mask
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1]))
        depth = rel_mag_grad(depth, 0.05)
        if self.with_figures and self.cfg.figure.sintel_debug:
            self.fig_sintel_debug.img_cnt += 1
            ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
            _ = ax.imshow(depth)
            ax.axis("equal")
            ax.axis("off")
        occ_mask, flow_warped = occlusion_mask(forward_flow, backward_flow, mask)
        if self.with_figures and self.cfg.figure.sintel_debug:
            self.fig_sintel_debug.img_cnt += 1
            ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
            # _ = ax.imshow(occ_mask)
            flow_warped = np.stack([flow_warped[:, :, 1], flow_warped[:, :, 0]], axis=-1)
            flow_warped_color = flow_vis.flow_to_color(flow_warped,
                                                       convert_to_bgr=False)
            _ = ax.imshow(flow_warped_color)
            ax.axis("equal")
            ax.axis("off")
        flow_boundary = np.logical_and(np.logical_or(depth, bnd), motion_chg)
        if self.with_figures and self.cfg.figure.sintel_debug:
            self.fig_sintel_debug.img_cnt += 1
            ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
            _ = ax.imshow(flow_boundary)
            ax.axis("equal")
            ax.axis("off")
        return flow_boundary, occ_mask

    def apply_sintel(self,
                     forward_flow,
                     backward_flow,
                     flow_pred,
                     flow_error,
                     segmentation,
                     depth,
                     mask):
        flow_boundary, occ_mask = self.find_motion_boundary(
            forward_flow, backward_flow, segmentation, depth, mask)
        boundary_size = np.sum(flow_boundary)
        # mask of bar
        mask2D = np.reshape(mask, (mask.shape[0], mask.shape[1]))
        mask_unoccluded = np.logical_and(mask2D, np.logical_not(occ_mask))
        error_matched, epe_matched = select_epe_region(flow_error, mask_unoccluded)
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(error_matched)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict['matched'] += epe_matched
        mask_unmatched = np.logical_and(mask2D, occ_mask)
        error_unmatched, epe_unmatched = select_epe_region(flow_error, mask_unmatched)
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(error_unmatched)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict['unmatched'] += epe_unmatched
        if boundary_size > 0:
            # Region d_close, d_medium, d_far
            d_close = border_distance(np.logical_not(flow_boundary), self.CLOSE_BND)
            d_close = d_close * mask_unoccluded
            res_d_close, epe_d_close = calc_epe(flow_pred, forward_flow, d_close)
            d_medium = border_distance(np.logical_not(
                flow_boundary), self.MEDIUM_BND, self.CLOSE_BND)
            d_medium = d_medium * mask_unoccluded
            res_d_medium, epe_d_medium = calc_epe(flow_pred, forward_flow, d_medium)
            d_far = border_distance(np.logical_not(
                flow_boundary), self.FAR_BND, self.MEDIUM_BND)
            d_far = d_far * mask_unoccluded
            res_d_far, epe_d_far = calc_epe(flow_pred, forward_flow, d_far)
        else:
            d_close = np.zeros(flow_boundary.shape)
            d_medium = d_close
            d_far = d_close
            epe_d_close = 0.0
            epe_d_medium = 0.0
            epe_d_far = 0.0
            if self.with_figures and self.cfg.figure.sintel:
                res_d_close = np.zeros(flow_boundary.shape)
                res_d_medium = np.zeros(flow_boundary.shape)
                res_d_far = np.zeros(flow_boundary.shape)
        if self.with_figures and self.cfg.figure.sintel_debug:
            self.fig_sintel_debug.img_cnt += 1
            ax = self.fig_sintel_debug.get_axis(self.fig_sintel_debug.img_cnt)
            _ = ax.imshow(d_close)
            ax.axis("equal")
            ax.axis("off")
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(res_d_close)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict[self.STR_CLOSE_BND] += epe_d_close
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(res_d_medium)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict[self.STR_MEDIUM_BND] += epe_d_medium
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(res_d_far)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict[self.STR_FAR_BND] += epe_d_far
        error_small, s_small = epe_select(
            flow_error, forward_flow, -1.0, self.SMALL_FLOW)
        self.row_dict[self.STR_SMALL_FLOW] += s_small
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(error_small)
            ax.axis("equal")
            ax.axis("off")
        error_medium, s_medium = epe_select(
            flow_error, forward_flow, self.SMALL_FLOW, self.MEDIUM_FLOW)
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(error_medium)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict[self.STR_MEDIUM_FLOW] += s_medium
        error_large, s_large = epe_select(
            flow_error, forward_flow, self.MEDIUM_FLOW)
        if self.with_figures and self.cfg.figure.sintel:
            self.fig_sintel.img_cnt += 1
            ax = self.fig_sintel.get_axis(self.fig_sintel.img_cnt)
            _ = ax.imshow(error_large)
            ax.axis("equal")
            ax.axis("off")
        self.row_dict[self.STR_LARGE_FLOW] += s_large
        return

    def process_movie(self, train_data):
        if self.csvfile is None:
            self.open_csv()

        self.video_name, self.video_type = get_video_names(train_data['metadata'])
        self.row_dict['video_name'] = self.video_name
        self.row_dict['video_type'] = self.video_type
        f_scale, f_offset = get_scale_offset(train_data['metadata'])
        bf_scale, bf_offset = get_scale_offset(train_data['metadata'], 'backward_flow')
        d_scale, d_offset = get_scale_offset(train_data['metadata'], 'depth')
        self.row_dict['epe'] = 0
        self.num_frames = int(train_data['metadata']['num_frames'])
        if self.cfg['num_frames'] > 0:
            self.num_frames = min(self.num_frames, self.cfg.num_frames)

        if self.with_figures:
            self.setup_figures()

        for i in range(self.num_frames - 1):
            # Plot input videos
            if self.with_figures and self.cfg.figure.images:
                self.fig_flow.img_cnt += 1
                ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                ax.imshow(train_data['video'][i, :, :, :])
                ax.axis("off")
                self.fig_flow.img_cnt += 1
                ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                ax.imshow(train_data['video'][i + 1, :, :, :])
                ax.axis("off")

            # Get mask for segmentation
            segmentation = train_data['segmentations'][i, :, :, :].numpy()
            mask = segmentation > 0  # Assuming any non-zero value is part of the mask
            fg_mask, bg_mask = get_fg_bg_mask(segmentation, mask)

            forward_flow = train_data['forward_flow'][i, :, :, :].numpy()
            forward_flow = f_scale * forward_flow + f_offset
            forward_flow = forward_flow * mask

            if self.with_figures and self.cfg.figure.images:
                # flip row and column as train_data is row,col but flow_vis expects column,row
                forward_flow_vis = np.stack([forward_flow[:, :, 1], forward_flow[:, :, 0]], axis=-1)
                flow_color = flow_vis.flow_to_color(forward_flow_vis, convert_to_bgr=False)
                self.fig_flow.img_cnt += 1
                ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                ax.imshow(flow_color)
                ax.axis("off")

            frame1 = train_data['video'][i, :, :, :].numpy()
            frame2 = train_data['video'][i + 1, :, :, :].numpy()
            flow_pred, _, _, _ = self.predictor.predict(frame1, frame2)
            flow_pred_vis = flow_pred * mask
            flow_pred = np.stack([flow_pred_vis[:, :, 1], flow_pred_vis[:, :, 0]], axis=-1)

            if self.with_figures and self.cfg.figure.images:
                predflow_color = flow_vis.flow_to_color(flow_pred_vis, convert_to_bgr=False)
                self.fig_flow.img_cnt += 1
                ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                ax.imshow(predflow_color)
                ax.axis("off")

                if self.cfg.figure.quiver:
                    self.fig_flow.img_cnt += 1
                    ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                    _, self.fig_flow.x_mesh, self.fig_flow.y_mesh = flow_quiver(
                        forward_flow, self.fig_flow.x_mesh, self.fig_flow.y_mesh, ax)
                    ax.axis("equal")
                    ax.axis("off")
                    self.fig_flow.img_cnt += 1
                    ax = self.fig_flow.get_axis(self.fig_flow.img_cnt)
                    _, self.fig_flow.x_mesh, self.fig_flow.y_mesh = flow_quiver(
                        flow_pred, self.fig_flow.x_mesh, self.fig_flow.y_mesh, ax)
                    ax.axis("equal")
                    ax.axis("off")

            # Calculate errors
            flow_error = self.calc_error(flow_pred, forward_flow, mask, fg_mask, bg_mask)
            # print(f"[test_flow] Frame {i} EPE: {flow_error.mean()}")

            # if False:  # 'jacobian':#self.cfg.error.mpi_sintel :
            #     depth = train_data['depth'][i, :, :, :].numpy()
            #     depth = d_scale * depth + d_offset
            #     backward_flow = train_data['backward_flow'][i + 1, :, :, :].numpy()
            #     backward_flow = bf_scale * backward_flow + bf_offset
            #     # do we need to mask the backward flow?
            #     self.apply_sintel(forward_flow, backward_flow, flow_pred, flow_error,
            #                       segmentation, depth, mask)

        try:
            self.write_stats()
        except:
            print("[test_flow]", "Could not write row", self.row_dict)

        if self.with_figures:
            self.save_figures()
        # In case the file is run as a script
        # plt.show()

    def run(self):
        dataset_info = tfds.builder(self.cfg.tfds + '/' + self.cfg.tfds_config).info
        available_splits = dataset_info.splits.keys()

        if self.cfg.split not in available_splits:
            # Use the first available split as a fallback
            print(f"[test_flow] 'train' split not found, skipping {self.cfg.tfds_config}...")
            return

        sample_count = len(self.flow_dataset[self.cfg.split])
        if sample_count > 0:
            print(f"\n[test_flow] Number of videos in {str(self.cfg.tfds_config)}: {sample_count}")
            m_iter = iter(self.flow_dataset[self.cfg.split])
            for _ in range(sample_count):
                flow_data = next(m_iter)
                self.process_movie(flow_data)
        else:
            for flow_data in self.flow_dataset[self.cfg.split]:
                self.process_movie(flow_data)
        return


def main():
    with open('/workspace/eval_conf/config.yaml') as f:
        dict_config = yaml.safe_load(f)

    dataset_base_path = os.path.join(dict_config['base'], dict_config['tfds'])
    print("Loading datasets in " + dataset_base_path)
    for dataset_folder in os.listdir(dataset_base_path):
        if not dataset_folder.startswith('.'):
            print(f"[test_flow] Processing dataset: {dataset_folder}")
            dict_config['tfds_config'] = dataset_folder
            TestFlow(cfg=dict_config).run()


if __name__ == '__main__':
    main()
