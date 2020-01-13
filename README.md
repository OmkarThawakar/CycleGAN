# CycleGAN
Cycle GAN implementation in tensorflow 2.0

## Requirnments

1. Tensorflow 2.0
2. Matplotlib


## Dataset

Dataset for training of CycleGAN can be downloaded with following code.
This dataset contain images of Horses and Zebras. 
Our goal is to generate zebras from horses from image to image translation with unpaired training.

```
_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'

path_to_zip = tf.keras.utils.get_file('horse2zebra.zip',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'dataset/')
```
## Train the Model

Windows
```
python cyclegan.py
```
Linux and MAC
```
CUDA_VISIBLE_DEVICES='GPU_NO' python cyclegan.py
```
## Test
In code change mode in **kwargs 

```
kwargs = {'epochs': 500, 
			'path': 'dataset',
				'mode':'test', 
				'output_path':'Exp_1',
				'batch_size':1,
	        }
 ```
 Run same command for testing
 ```
 python cyclegan.py
 ```
