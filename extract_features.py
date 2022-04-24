import warnings

import numpy as np
import pandas as pd
from pydub.utils import mediainfo

from inaSpeechSegmenter import Segmenter

warnings.filterwarnings("ignore", module='pyannote')


def segment_audio(media_file_path):
    """Segment audio using inaSpeechSegmenter CNN"""

    # Perform segmentation
    # create an instance of speech segmenter
    # this loads neural networks and may last few seconds
    seg = Segmenter(detect_gender=False)

    # segmentation is performed using the __call__ method of the segmenter instance
    # the result is a list of tuples
    # each tuple contains:
    # * label in 'speech', 'music', 'noEnergy'
    # * start time of the segment
    # * end time of the segment
    segmentation = seg(media_file_path)

    # Save & format the result
    # put the result in a dataframe
    time_points = np.array(segmentation)[:, 1:].astype(float).round(2)
    labels = np.array(segmentation)[:, 0]

    df_segments_ = pd.DataFrame(np.c_[time_points[:, 0].round(2), time_points[:, 1].round(
        2), labels], columns=['start', 'stop', 'label'])
    df_segments_[['start', 'stop']] = df_segments_[['start', 'stop']].astype('float')

    # add the duration of each segment as a column
    df_segments_['duration'] = df_segments_.stop - df_segments_.start
    df_segments_ = df_segments_[['start', 'stop', 'duration', 'label']]
    df_segments_['label'] = df_segments_['label'].str.replace('noEnergy', 'silence')

    # Calculate the percentage of each label
    call_duration = float(mediainfo(media_file_path)['duration'])
    labels = ["speech", "silence", "noise", "music"]

    call_stats_ = {}
    for label in labels:
        duration = df_segments_[df_segments_['label'] == label].duration.sum()
        percentage = np.round(100 * duration / call_duration, 2)
        call_stats_[label] = percentage
    call_stats_['silence'] = call_stats_.pop('silence')
    call_stats_["speech"] = call_stats_['speech']

    return call_stats_, df_segments_


if __name__ == 'main':
    call_stats, df_call_segments = segment_audio("20160102_102856.gsm")
    print(call_stats)
    print(df_call_segments)
