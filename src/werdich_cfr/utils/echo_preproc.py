import numpy as np
import cv2
import lz4.frame

class Echoproc:
    """ Preprocessing functions for echo videos """

    def __init__(self, min_rate, min_frames):
        self.min_rate = min_rate
        self.min_frames = min_frames

    def im_scale(self, im, dx, dy):
        """ convert single images to uint8 and resize by scale factors """
        # We can do other things here: e.g. background subtraction or contrast enhancement
        im_scaled = np.uint8((im - np.amin(im)) / (np.amax(im) - np.amin(im)) * 256)
        # im_scaled_eq = cv2.equalizeHist(im_scaled) # histogram equalization (not needed)
        if (dx is not None) & (dy is not None):
            width = int(np.round(im_scaled.shape[1] * 7.5 * dx))
            height = int(np.round(im_scaled.shape[0] * 7.5 * dy))
            im_resized = cv2.resize(im_scaled, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            im_resized = im_scaled
        return im_resized

    def data2imarray(self, im_data, dx=None, dy=None):
        """
        apply imscale function to np.array
        arg: im_array (frame, height, width)
        returns: im_array (height, width, frame)
        """
        im_data = np.squeeze(im_data)
        im_list = [self.im_scale(im_data[im], dx, dy) for im in range(im_data.shape[0])]
        im_array = np.array(im_list, dtype=np.uint16)
        im_array = np.moveaxis(im_array, 0, -1)
        return im_array

    def subsample_time_index_list(self, frame_time, default_rate, n_frames):
        """
        frame_time: time interval between frames [s]
        default_rate: matching frame rate [fps],
        n_frames: number of frames in the output
        """
        default_times = np.arange(0, n_frames, 1) / default_rate
        times = np.arange(0, default_times[-1] + frame_time, frame_time)
        time_index_list = [np.argmin(np.abs(times - t)) for t in default_times]

        return time_index_list

    def subsample_video(self, image_array, frame_time, min_rate, min_frames):
        """
        Select frames that are closest to a constant frame rate
        arg: image_array: np.array() [rows, columns, frame]
        """
        convert_video = True
        rate = 1 / frame_time
        # Check if the video is long enough
        min_video_len = min_frames / min_rate
        video_len = image_array.shape[-1] / rate
        if (min_video_len <= video_len) & (min_rate < rate):
            # print('Video is long enough and the rate is good.')
            # Get the frame index list
            time_index_list = self.subsample_time_index_list(frame_time=frame_time,
                                                        default_rate=min_rate,
                                                        n_frames=min_frames)
            # Select the frames from the video
            image_array = image_array[:, :, time_index_list]
        else:
            convert_video = False

        return convert_video, image_array

    def load_video(self, ser):
        """ Load video into memory from metadata in dataframe
        ser.filename
        ser.dir
        """

        file = os.path.join(ser.dir, filename)

        try:
            with lz4.frame.open(file, 'rb') as fp:
                data = np.load(fp)

        except IOError as err:
            print('Could not open this file: {}\n {}'.format(file, err))
        else:
            im_array_original = self.data2imarray(data, dx=ser.deltaX, dy=ser.deltaY)
            frame_time = ser.frame_time * 1e-3
            convert_video, im_array = subsample_video(image_array=im_array_original,
                                                      frame_time=frame_time,
                                                      min_rate=min_rate,
                                                      min_frames=min_frames)
