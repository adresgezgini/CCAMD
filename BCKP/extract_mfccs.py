import os
import pickle
import urllib
from urllib.parse import urlparse

import librosa
import pandas as pd


def file_download(in_):
    """
    This function downloads an audio file
    from the AdresGezgini FTP server.
    It returns the path to the file on the hard drive

    in_ is a pandas dataframe
    """
    # create directory for processing temporary files if it doesn't exist
    savepath = '.processed/'
    try:
        # Create target Directory
        os.mkdir(savepath)
    except FileExistsError:
        pass

    print(in_)
    if in_['CallType'] == 0:
        the_link = 'https://callcenter.adresgezgini.com:5130/monitor/backup/outbound/' + in_['FilePath']
    elif in_['CallType'] == 1:
        the_link = 'https://callcenter.adresgezgini.com:5130/monitor/backup/inbound/' + in_['FilePath']
    elif in_['CallType'] == 2:
        the_link = 'https://callcenter.adresgezgini.com:5130/monitor/backup/conf/' + in_['FilePath']

    filename_ = os.path.split(urlparse(the_link).path)[1]
    filename_ = urllib.request.urlretrieve(the_link, f'{savepath}{filename_}')[0]

    return filename_


if  __name__ == '__main__':
    # read the data
    data = pd.read_csv("test_data_.csv")

    mfccs = []
    for index, item in data.iterrows():
        print(item)
        filename = file_download(item)

        y, sr = librosa.load(filename)

        mfccs.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
        os.remove(filename)

    with open('mfccs.pkl', 'wb') as f:
        pickle.dump(mfccs, f)

    os.rmdir('.processed/')    # remove the temporary directory    