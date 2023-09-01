## Dataset preparation
You should download the dataset from [urlhere] and unzip the dataset folder to the root directory of repository.

Then, run the following code to prepare all data needed.
```shell
cd code
python prepare_data.py --all --u-net
```
All processed data will take ~150 GiB space. You can adjust the configuration in conf.py to save space usage (e.g. set SAVE_SPECTROGRAM to False.)

## Training
To train models, use following code
```shell
python train.py --<model_name>
```
```model_name``` is ```u-net```, ```wave-u-net```, ```rnn``` or ```fullsubnet```

The trained model will be stored under ```models/``` folder.

## Testing
To test model, similarly:
```shell
python test.py --<model_name>
```
Test output will be put under model folder, SI-SDR result is in SDRMatrix.npy file, use ```np.load``` to load it into memory.
## Credits

#### Wave-u-net implementation:
Implementation of original wave-u-net is from: https://github.com/satvik-venkatesh/Wave-U-net-TF2

Modification is mine.

#### U-net implementation:
Implementation of original u-net is from: https://github.com/Zhz1997/Singing-voice-speration-with-U-Net

Modification is mine.

#### SI-SDR implementation:
https://github.com/aliutkus/speechmetrics/blob/master/speechmetrics/relative/sisdr.py
