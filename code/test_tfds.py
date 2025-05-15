import flow_vis
from postprocess_fisheye import apply_fisheye_to_flow
from postprocess_blocker import apply_blocker_mask
from tfds_util import *
import tensorflow_datasets as tfds


def plot_flows(train_data):
    # print the video info
    print('\n', train_data['instances']['asset_id'])
    seg = train_data['segmentations'][0, :, :, :].numpy()
    print("Min seg: " + str(np.min(seg)))
    print("Max seg: " + str(np.max(seg)))

    # retrieve video metadata and flow scale/offset
    video_name, video_type = get_video_names(train_data['metadata'])
    f_scale, f_offset = get_scale_offset(train_data['metadata'])

    num_frames = int(train_data['metadata']['num_frames'])  # get the number of frames in the video
    x_mesh = np.empty(1)
    y_mesh = np.empty(1)

    n_cols = 5  # number of columns for subplot grid

    fig = plt.figure(figsize=(n_cols, num_frames))
    plt.subplots_adjust(wspace=0, hspace=0)

    # loop through frames in the video
    for i in range(num_frames - 1):
        img_cnt = i * n_cols

        # plot the current frame
        plt.subplot(num_frames, n_cols, img_cnt + 1)
        plt.imshow(train_data['video'][i, :, :, :])
        plt.axis("off")
        if i == 0:
            plt.title("Frame N", fontsize=5)

        # plot the next frame
        plt.subplot(num_frames, n_cols, img_cnt + 2)
        plt.imshow(train_data['video'][i + 1, :, :, :])
        plt.axis("off")
        if i == 0:
            plt.title("Frame N+1", fontsize=5)

        seg = train_data['segmentations'][i, :, :, :].numpy()
        plt.subplot(num_frames, n_cols, img_cnt + 3)
        plt.imshow(seg, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Segmentation", fontsize=5)

        # forward optical flow for the current frame
        forward_flow = train_data['forward_flow'][i, :, :, :].numpy()
        print(f"Forward flow raw min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")

        forward_flow = f_scale * forward_flow + f_offset
        # print(f"Forward flow scaled min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")

        #forward_flow = forward_flow * (seg < np.max(seg) + 1)  # Use the segmentation mask
        #print(f"Forward flow masked min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")
        
        forward_flow = apply_blocker_mask(forward_flow,"/home/ethan/Documents/Niloofar/Projects/flowdataset/cam_textures/blocker.png")
        print(f"Forward flow Blocker min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")
        
        forward_flow = apply_fisheye_to_flow(forward_flow, 0.05, (train_data['metadata']['height'], train_data['metadata']['width']))
        print(f"Forward flow fisheye min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")

        # normalize the flow vectors
        mag = np.sqrt(forward_flow[:, :, 0] ** 2 + forward_flow[:, :, 1] ** 2)
        max_mag = np.max(mag)
        if max_mag > 0:
            forward_flow[:, :, 0] /= max_mag
            forward_flow[:, :, 1] /= max_mag
        # print(f"Forward flow normalized min: {np.min(forward_flow)}, max: {np.max(forward_flow)}")

        # optical flow
        forward_flow_vis = np.stack([forward_flow[:, :, 1], forward_flow[:, :, 0]], axis=-1)
        flow_color = flow_vis.flow_to_color(forward_flow_vis, convert_to_bgr=False)
        plt.subplot(num_frames, n_cols, img_cnt + 4)
        plt.imshow(flow_color)
        if i == 0:
            plt.title("Optical Flow", fontsize=5)
        plt.axis("off")

        # flow vectors using quiver plot
        ax = plt.subplot(num_frames, n_cols, img_cnt + 5)
        _, x_mesh, y_mesh = flow_quiver(forward_flow, x_mesh, y_mesh, ax=ax)
        plt.axis("equal")
        if i == 0:
            plt.title("Quiver", fontsize=5)
        plt.axis("off")

    save_fig(fig, video_name + "_" + video_type)  # save with the video name and type

    # in case the file is run as a script
    plt.show()


if __name__ == '__main__':
    # load a specific dataset from tfds
    flow_dataset = tfds.load('flow_dataset_builder/fixed_random_rotate_bar')

    for video_index, train_data in enumerate(flow_dataset['train']):
        if video_index >= 3:  # Iterate over the dataset to process the first few videos
            break
        print(f"==============================")
        print(f"Processing video {video_index}")
        plot_flows(train_data)
