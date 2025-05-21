Please check the original repo: https://github.com/lizhe00/AnimatableGaussians

This version provides the scripts that will benchmark the submissions.

### Installation

Same steps as the original repo: clone the repo, install the requirements, then `python setup.py install` for gaussians/diff_gaussian_rasterization_depth_alpha and network/styleunet.
Unfortunately, since doing just `pip install -r requirements.txt` fails more often than not, here is a list of commands that *should* work fine (provided you just created your venv, for example with python 3.12):
```
pip install --upgrade pip
pip install torch
pip install torchvision
pip install setuptools
pip install wheel
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install opencv-python
pip install trimesh
pip install scikit-image
pip install plyfile
pip install pytorch-fid
```

### Complementary files

Here is the tricky part.

* You will first need the [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) model.
Place the files to ./smpl_files/smplx, so that it looks like this:
  * smpl_files
    * mano
      * ...
    * smplx
      * SMPLX_FEMALE.npz
      * SMPLX_FEMALE.pkl
      * etc.
* To run the basic example, you will need the pretrained models for [zzr](https://drive.google.com/file/d/1lR_O9m0J_lwc8POA_UtCDM9LsTWOIu4m/view?usp=sharing), [lbn1](https://drive.google.com/file/d/1P-s-RcJ5_Z7ZVSzjjl-xhPCExqN8td7S/view?usp=sharing) and [lbn2](https://drive.google.com/file/d/1KakiePoLpV3Wa0QGtnzrt8MAhZbNQi6n/view?usp=sharing).
These files contain a net.pt file, for example at avatarrex_zzr/batch_700000/net.pt; rename them as net_zzr.pt, net_lbn1.pt and net_lbn2.pt, and place them in model_example, such that it looks like this:
  * model
    * example
      * load_model.py
      * net_lbn1.pt
      * net_lbn2.pt
      * net_zzr.pt
* You will need to download the AvatarRex dataset, and more precisely the zzr, lbn1 and lbn2 avatars.
The delicate part is that you will need to combine files from both the original datasets [zzr](https://drive.google.com/file/d/1sCQJ3YU-F3lY9p_HYNIQbT7QyfVKy0HT/view?usp=sharing), 
[lbn1](https://drive.google.com/file/d/1DuESdA5YwvJKapyo7i_KoQxKHHFWzi-w/view?usp=sharing) and [lbn2](https://drive.google.com/file/d/1J7ITsYhuWlqhoIkmYni8dL2KJw-wmcy_/view?usp=sharing)
and the preprocessed datasets [zzr](https://drive.google.com/file/d/1lR_O9m0J_lwc8POA_UtCDM9LsTWOIu4m/view?usp=sharing), [lbn1](https://drive.google.com/file/d/1P-s-RcJ5_Z7ZVSzjjl-xhPCExqN8td7S/view?usp=sharing) and [lbn2](https://drive.google.com/file/d/1KakiePoLpV3Wa0QGtnzrt8MAhZbNQi6n/view?usp=sharing).
The preprocessed datasets should provide the /smpl_pos_map folder and the cano_weight_volume.npz file; 
the original dataset, should provide the calibration_full.json and the smpl_params.npz files, as well as the original raw images and masks (divided into folders with names such as 22010708, each corresponding to a different camera).
As stated in the original repo, **using this dataset is bound to agreements**, that are detailed in the original repo's [dedicated README](https://github.com/lizhe00/AnimatableGaussians/blob/master/AVATARREX_DATASET.md).
Once everything is downloaded, create a /data folder and place the files like this:
    * data
      * avatarrex
        * avatarrex_lbn1
          * 22010708
          * 22010710
          * ...
          * 22139907
          * smpl_pos_map
            * ...
          * calibration_full.json
          * cano_weight_volume.npz
          * smpl_params.npz
        * avatarrex_lbn2
          * 22010708
          * 22010710
          * ...
          * 22139907
          * smpl_pos_map
            * ...
          * calibration_full.json
          * (no cano_weight_volume.npz for whatever reason...)
          * smpl_params.npz
        * avatarrex_zzr
          * 22010708
          * 22010710
          * ...
          * 22139907
          * smpl_pos_map
              * ...
          * calibration_full.json
          * cano_weight_volume.npz
          * smpl_params.npz
          
### How to use

Once the previous steps are done, if I did not mess up anything, everything should work fine, and you can use the launch_example_benchmark.sh script to launch a demo without encountering problems.

The main python script to understand is main_avatar.py.
It uses three simple arguments:
* --model_name: all "models" are stored in the /model folder. A "model" is actually a folder itself, that contains at least a load_model.py file (this will be further explained in the **Instructions** section).
The "model name" itself is therefore the name of the folder that contains this file, as well as your whole submission. This way, one can store multiple submissions under /model and select which one to run with this argument.
* --avatar: choice between "zzr", "lbn1" and "lbn2".
* --camera: choice between 7 and 13. Those two cameras were excluded, as a test set, from the training of the default pretrained models, and are therefore the ones that will be used for benchmarking.

Therefore, the command `python main_avatar.py --model_name example --avatar zzr --camera 7` should, for the zzr avatar, generate images according to the camera 7, using the model in /model/example, and then compute a bunch of metrics and store them as ./example_zzr_7.json.

### Instructions

Your submission should consist in a folder, containing at least one file named load_model.py.
This python file should contain, such as in the example, a function `load_model(avatar, device)`, with:
* avatar: a string between "zzr", "lbn1" and "lbn2"
* device: such as "cpu" or "cuda"

This function is the one to be loaded by main_avatar.py, and that will instantiate your whole AvatarNet, that is responsible for the rendering and includes the various neural networks.
This way, you should be able to plug freely your own AvatarNet, as long as it is self-contained enough.

#### Warning

For every avatar, the camera 7 and 13 were excluded from training as a test set; please do not cheat by training on the test set.