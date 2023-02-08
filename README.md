# De-anonymization through face recognition on PET data &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/SebastianPartarrieu/live-kinect/blob/master/LICENSE)

It is fairly well known that CT and MRI scans, though de-identified, can potentially be re-identified by reconstructing the faces of the patients and using powerful facial recognition software. This is a privacy and security hazard as many public datasets have provided de-identified patient scans without adequately modifying patient facial features to avoid potential breaches. Although this is changing with specific 'de-facing' software now being provided for CT and MRI scans, the issue remains largely unexplored for PET scans. At first, it seems that due to the inherent lower resolution of PET imaging there might be no problem to address as any facial reconstruction will be too noisy to exploit.

In this repository, we explore the potential methods that might be employed to try and de-anonymize pet images through morphological reconstruction and face recognition. We will compare results to those obtained with the associated CT scans.

![Face detection](./docs/face_id_woman1.png)

![Facial landmarks](./docs/face_id_woman2.png)

In an ideal setting, we would have access to some of the patient's real-life photos to try and perform the matching ourselves and quantify re-identification performance, however we do not have our own patient cohort. As a substitute, we use metrics such as the number of patients where we can accurately locate the face using standard face detection algorithms, or those where we can accurately place facial landmarks which are a key component of most face detection software. Finally, comparing to CT scan face reconstructions provides further quantification of our approaches.

## Installation

### Getting started
Create a virtual environment using your [favorite method](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) (venv, conda, etc...) but make sure to install the dependencies using the provided environment.yml file.
If you feel like using conda `conda env create -f environment.yml` should do the trick. Note that most of the code was run remotely using Colab, so you don't necessarily need to run a local install, you can just access the notebooks directly on Colab.

### Download datasets
Follow instruction details [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287#932582870c7caa21e8b840a393398eeda1279f3b). Note that there is a substantial amount of data (~350Gb of data) so make sure you have enough space.

### System requirements
Most of the code here was run on Colab, any working python installation >=3.8 should do the trick.

## Experiment details

### Reconstructing faces
The main pipeline follows: 3D voxel representation -> adaptive thresholding to keep relevant voxels -> selecting largest connected component -> generating mesh representations of isosurface using marching cubes algorithm -> ray casting to get a 2D image representation of a patient's face -> applying face detection and landmark placement algorithms for potential de-anonymization.

### Denoising PET data

#### Traditional approaches
TODO
#### Deep-learning based
TODO

### Facial recognition
- Face detection is performed with [OpenCV Haar Feature-based Cascade Classifiers](https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html). This works well on grayscale images and though it is dated, it remains easy to implement and obtains decent results. 
- Landmark placement to create a Face Mesh is done using Google's [Mediapipe](https://google.github.io/mediapipe/). We need to be careful as these pipelines are tuned for RGB images and we are working with grayscale.


### File details
TODO
