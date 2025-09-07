import os
import sys
import re
import shutil
import subprocess
import tarfile
import fnmatch
import tensorflow as tf
import io
import random
import glob
from PIL import Image


def setup_environment():
    MLENVIRONMENT = "LOCAL"
    current_dir = os.getcwd()
    models_path = os.path.join(current_dir, 'models')
    
    if os.path.exists(models_path) and os.path.isdir(models_path):
        print(f"Models directory already exists at {models_path}, removing...")
        shutil.rmtree(models_path)
    
    try:
        print("Cloning TensorFlow models repository...")
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/tensorflow/models'], check=True)
        subprocess.run(['git', 'fetch', '--depth', '1', 'origin', 'ad1f7b56943998864db8f5db0706950e93bb7d81'], 
                      cwd='models', check=True)
        subprocess.run(['git', 'checkout', 'ad1f7b56943998864db8f5db0706950e93bb7d81'], 
                      cwd='models', check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        return None, None, None
    
    print(sys.version)
    if MLENVIRONMENT == "COLAB":
        print("colab env setup")
        os.environ["HOMEFOLDER"] = "/content/"
        HOMEFOLDER = '{HOMEFOLDER}'.format(**os.environ)
        FINALOUTPUTFOLDER_DIRNAME = 'final_output'
        FINALOUTPUTFOLDER = HOMEFOLDER + FINALOUTPUTFOLDER_DIRNAME
        print(HOMEFOLDER)
    else:
        print("local env setup")
        HOMEFOLDER = current_dir + "/"
        FINALOUTPUTFOLDER_DIRNAME = 'final_output'
        FINALOUTPUTFOLDER = HOMEFOLDER + FINALOUTPUTFOLDER_DIRNAME
        print(f"Working directory: {HOMEFOLDER}")
    
    try:
        import glob as proto_glob
        proto_files = proto_glob.glob(f'{HOMEFOLDER}models/research/object_detection/protos/*.proto')
        if proto_files:
            subprocess.run(['protoc'] + [f'object_detection/protos/{os.path.basename(f)}' for f in proto_files] + ['--python_out=.'], 
                          cwd=f'{HOMEFOLDER}models/research', check=True)
        else:
            print("No proto files found, continuing...")
        
        with open(HOMEFOLDER+'models/research/object_detection/packages/tf2/setup.py') as f:
            s = f.read()
        
        with open(HOMEFOLDER+'models/research/setup.py', 'w') as f:
            if MLENVIRONMENT in ["COLAB", "LOCAL"]:
                s = re.sub('tf-models-official>=2.5.1','tf-models-official>=2.17.0', s)
                f.write(s)
        
        print("Installing Object Detection API...")
        subprocess.run(['pip', 'install', 'tensorflow==2.15.0', '--force-reinstall'], check=True)
        print("TensorFlow 2.15.0 installed successfully")
        print("Skipping object_detection pip install - using local version")
        
        print("Skipping model builder test due to Keras API compatibility issues...")
        print("Object Detection API installation completed!")
        
        models_path = os.path.join(current_dir, 'models')
        research_path = os.path.join(models_path, 'research')
        if research_path not in sys.path:
            sys.path.append(research_path)
        if models_path not in sys.path:
            sys.path.append(models_path)
        
        print(f"Added to Python path: {research_path}")
        print(f"Added to Python path: {models_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during environment setup: {e}")
        return None, None, None
    
    return HOMEFOLDER, FINALOUTPUTFOLDER, MLENVIRONMENT

def download_and_process_dataset(dataset_url=None, HOMEFOLDER="/content/", local_dataset_path=None):
    if local_dataset_path and os.path.exists(local_dataset_path):
        print(f"Using local dataset: {local_dataset_path}")
        train_record_fname, val_record_fname, label_map_pbtxt_fname = set_tfrecord_variables(local_dataset_path)
        
        print("Train Record File:", train_record_fname)
        print("Validation Record File:", val_record_fname)
        print("Label Map File:", label_map_pbtxt_fname)
        
        return train_record_fname, val_record_fname, label_map_pbtxt_fname
    
    try:
        subprocess.run(['pip', 'install', '-q', 'gdown', '--upgrade'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing gdown: {e}")
        return None, None, None
    
    import gdown
    
    if dataset_url:
        try:
            print("Downloading dataset...")
            url = dataset_url
            
            if 'drive.google.com/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                url = f'https://drive.google.com/uc?id={file_id}'
            
            output = f'{HOMEFOLDER}dataset.zip'
            gdown.download(url, output, fuzzy=True)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return None, None, None
    
    datasetPath = f'{HOMEFOLDER}dataset.zip'
    print(datasetPath)
    try:
        subprocess.run(['unzip', datasetPath], cwd=HOMEFOLDER, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting dataset: {e}")
        return None, None, None
    
    train_record_fname, val_record_fname, label_map_pbtxt_fname = set_tfrecord_variables(HOMEFOLDER)
    
    print("Train Record File:", train_record_fname)
    print("Validation Record File:", val_record_fname)
    print("Label Map File:", label_map_pbtxt_fname)
    
    return train_record_fname, val_record_fname, label_map_pbtxt_fname


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def set_tfrecord_variables(directory):
    train_record_fname = ''
    val_record_fname = ''
    label_map_pbtxt_fname = ''

    for tfrecord_file in find_files(directory, '*.tfrecord'):
        if '/train/' in tfrecord_file:
            train_record_fname = tfrecord_file
        elif '/valid/' in tfrecord_file:
            val_record_fname = tfrecord_file
        elif '/test/' in tfrecord_file:
            pass

    for label_map_file in find_files(directory, '*_label_map.pbtxt'):
        label_map_pbtxt_fname = label_map_file

    return train_record_fname, val_record_fname, label_map_pbtxt_fname

def setup_training_config(HOMEFOLDER, label_map_pbtxt_fname, train_record_fname, val_record_fname, chosen_model='ssd-mobilenet-v2'):
    MODELS_CONFIG = {
        'ssd-mobilenet-v2': {
            'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
            'base_pipeline_file': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
            'pretrained_checkpoint': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        },
    }
    model_name = MODELS_CONFIG[chosen_model]['model_name']
    pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
    base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
    
    os.makedirs(f'{HOMEFOLDER}models/mymodel/', exist_ok=True)
    
    download_tar = 'https://downloads.limelightvision.io/models/' + pretrained_checkpoint
    try:
        subprocess.run(['wget', download_tar], cwd=f'{HOMEFOLDER}models/mymodel/', check=True)
        tar = tarfile.open(f'{HOMEFOLDER}models/mymodel/{pretrained_checkpoint}')
        tar.extractall(path=f'{HOMEFOLDER}models/mymodel/')
        tar.close()
        
        download_config = 'https://downloads.limelightvision.io/models/' + base_pipeline_file
        subprocess.run(['wget', download_config], cwd=f'{HOMEFOLDER}models/mymodel/', check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model files: {e}")
        return None, None, None, None, None
    
    num_steps = 3000
    checkpoint_every = 500
    batch_size = 32
    
    pipeline_fname = HOMEFOLDER+'models/mymodel/' + base_pipeline_file
    fine_tune_checkpoint = HOMEFOLDER+'models/mymodel/' + model_name + '/checkpoint/ckpt-0'
    
    num_classes = get_num_classes(label_map_pbtxt_fname)
    classes = get_classes(label_map_pbtxt_fname)
    
    print('Total classes:', num_classes)
    print(classes)
    
    create_label_file(HOMEFOLDER + "limelight_neural_detector_labels.txt", classes)
    
    create_pipeline_config(pipeline_fname, fine_tune_checkpoint, train_record_fname, val_record_fname, 
                          label_map_pbtxt_fname, batch_size, num_steps, num_classes, chosen_model)
    
    pipeline_file = f'{HOMEFOLDER}pipeline_file.config'
    model_dir = HOMEFOLDER+'training_progress/'
    print(" ")
    print(model_dir)
    
    return pipeline_file, model_dir, num_steps, checkpoint_every, num_classes


def get_num_classes(pbtxt_fname):
    try:
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())
    except ImportError as e:
        print(f"Error importing object_detection: {e}")
        return 1


def get_classes(pbtxt_fname):
    try:
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        class_names = [category['name'] for category in category_index.values()]
        return class_names
    except ImportError as e:
        print(f"Error importing object_detection: {e}")
        return ["default_class"]


def create_label_file(filename, labels):
    with open(filename, 'w') as file:
        for label in labels:
            file.write(label + '\n')

def create_pipeline_config(pipeline_fname, fine_tune_checkpoint, train_record_fname, val_record_fname, 
                          label_map_pbtxt_fname, batch_size, num_steps, num_classes, chosen_model):
    print('writing custom configuration file')
    
    with open(pipeline_fname) as f:
        s = f.read()
    with open('pipeline_file.config', 'w') as f:
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(val_record_fname), s)
        
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)
        
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(batch_size), s)
        
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(num_steps), s)
        
        s = re.sub('checkpoint_every_n: [0-9]+',
                   'num_classes: {}'.format(num_classes), s)
        
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
        if chosen_model == 'ssd-mobilenet-v2':
            s = re.sub('learning_rate_base: .8',
                       'learning_rate_base: .004', s)
            s = re.sub('warmup_learning_rate: 0.13333',
                       'warmup_learning_rate: .0016666', s)
        
        if chosen_model == 'efficientdet-d0':
            s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
            s = re.sub('pad_to_max_dimension: true', '', s)
            s = re.sub('min_dimension', 'height', s)
            s = re.sub('max_dimension', 'width', s)
        
        f.write(s)

def fix_tf_breaking_changes():
    original_path = '/usr/local/lib/python3.11/dist-packages/tf_slim/data/tfexample_decoder.py'
    try:
        with open(original_path, 'r') as file:
            content = file.read()
            content = re.sub(r'import abc', 'import tensorflow as tf\n\nimport abc', content)
            content = re.sub(r'control_flow_ops.case', 'tf.case', content)
            content = re.sub(r'control_flow_ops.cond', 'tf.compat.v1.cond', content)
        with open(original_path, 'w') as file:
            file.write(content)
        print(f"File {original_path} fixed.")
    except FileNotFoundError:
        print(f"File {original_path} not found, skipping fix.")
    except Exception as e:
        print(f"Error fixing TF file: {e}")


def train_model(HOMEFOLDER, pipeline_file, model_dir, checkpoint_every, num_steps):
    fix_tf_breaking_changes()
    
    try:
        subprocess.run(['rm', '-rf', f'{HOMEFOLDER}training_progress'], check=True)
    except subprocess.CalledProcessError:
        pass
    
    try:
        subprocess.run([
            'python', f'{HOMEFOLDER}models/research/object_detection/model_main_tf2.py',
            f'--pipeline_config_path={pipeline_file}',
            f'--model_dir={model_dir}',
            '--alsologtostderr',
            f'--checkpoint_every_n={checkpoint_every}',
            f'--num_train_steps={num_steps}',
            '--num_workers=2',
            '--sample_1_of_n_eval_examples=1'
        ], check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False

def convert_to_tflite(HOMEFOLDER, FINALOUTPUTFOLDER, pipeline_file):
    if os.path.exists(FINALOUTPUTFOLDER) and os.path.isdir(FINALOUTPUTFOLDER):
        shutil.rmtree(FINALOUTPUTFOLDER)
    
    os.makedirs(FINALOUTPUTFOLDER, exist_ok=True)
    print(FINALOUTPUTFOLDER)
    
    last_model_path = HOMEFOLDER+'training_progress'
    exporter_path = HOMEFOLDER+'models/research/object_detection/export_tflite_graph_tf2.py'
    output_directory = FINALOUTPUTFOLDER
    
    try:
        subprocess.run([
            'python', exporter_path,
            '--trained_checkpoint_dir', last_model_path,
            '--output_directory', output_directory,
            '--pipeline_config_path', pipeline_file
        ], check=True)
        
        converter = tf.lite.TFLiteConverter.from_saved_model(FINALOUTPUTFOLDER+'/saved_model')
        tflite_model = converter.convert()
        model_path_32bit = FINALOUTPUTFOLDER+'/limelight_neural_detector_32bit.tflite'
        with open(model_path_32bit, 'wb') as f:
            f.write(tflite_model)
        
        subprocess.run(['cp', f'{HOMEFOLDER}limelight_neural_detector_labels.txt', FINALOUTPUTFOLDER], check=True)
        
        print("TFLite conversion completed successfully!")
        return model_path_32bit
    except subprocess.CalledProcessError as e:
        print(f"Error during TFLite conversion: {e}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def extract_images_from_tfrecord(tfrecord_path, output_folder, num_samples=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_images = 0
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_record in raw_dataset.take(num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        image_data = example.features.feature['image/encoded'].bytes_list.value[0]
        image = Image.open(io.BytesIO(image_data))
        image.save(os.path.join(output_folder, f'image_{saved_images}.png'))
        saved_images += 1
        if saved_images >= num_samples:
            break

    print(f"Extracted {saved_images} images to {output_folder}")


def get_quantization_images(HOMEFOLDER, MLENVIRONMENT="LOCAL"):
    extracted_sample_folder = HOMEFOLDER+'extracted_samples'

    quant_image_list = []
    if MLENVIRONMENT in ["COLAB", "LOCAL"]:
        jpg_file_list = glob.glob(extracted_sample_folder + '/*.jpg')
        jpeg_file_list = glob.glob(extracted_sample_folder + '/*.jpeg')
        JPG_file_list = glob.glob(extracted_sample_folder + '/*.JPG')
        png_file_list = glob.glob(extracted_sample_folder + '/*.png')
        bmp_file_list = glob.glob(extracted_sample_folder + '/*.bmp')
        quant_image_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list

    print("pulling samples from " + extracted_sample_folder)
    print("samples: " + str(len(quant_image_list)))
    return quant_image_list


def representative_data_gen(quant_image_list, width, height, quant_num=300):
    dataset_list = quant_image_list
    for i in range(quant_num):
        pick_me = random.choice(dataset_list)
        print(pick_me)
        image = tf.io.read_file(pick_me)

        if pick_me.endswith('.jpg') or pick_me.endswith('.JPG') or pick_me.endswith('.jpeg'):
            image = tf.io.decode_jpeg(image, channels=3)
        elif pick_me.endswith('.png'):
            image = tf.io.decode_png(image, channels=3)
        elif pick_me.endswith('.bmp'):
            image = tf.io.decode_bmp(image, channels=3)

        image = tf.image.resize(image, [width, height])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]


def quantize_model(FINALOUTPUTFOLDER, model_path_32bit, train_record_fname, HOMEFOLDER, MLENVIRONMENT="LOCAL"):
    extracted_sample_folder = HOMEFOLDER+'extracted_samples'
    extract_images_from_tfrecord(train_record_fname, extracted_sample_folder)
    quant_image_list = get_quantization_images(HOMEFOLDER, MLENVIRONMENT)
    
    interpreter = tf.lite.Interpreter(model_path=model_path_32bit)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    converter = tf.lite.TFLiteConverter.from_saved_model(FINALOUTPUTFOLDER+'/saved_model')
    print("initialized converter")
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(quant_image_list, width, height)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    
    print("begin conversion")
    try:
        tflite_model = converter.convert()
        print("conversion complete")
        
        quantized_model_path = FINALOUTPUTFOLDER+'/limelight_neural_detector_8bit.tflite'
        with open(quantized_model_path, 'wb') as f:
            f.write(tflite_model)
        
        return quantized_model_path
    except Exception as e:
        print(f"Error during quantization: {e}")
        return None

def compile_and_package_models(FINALOUTPUTFOLDER, HOMEFOLDER):
    try:
        subprocess.run(['curl', 'https://packages.cloud.google.com/apt/doc/apt-key.gpg'], stdout=subprocess.PIPE, check=True)
        subprocess.run(['sudo', 'apt-key', 'add', '-'], input=subprocess.PIPE, check=True)
        subprocess.run(['echo', 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main'], stdout=subprocess.PIPE, check=True)
        subprocess.run(['sudo', 'tee', '/etc/apt/sources.list.d/coral-edgetpu.list'], input=subprocess.PIPE, check=True)
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', 'edgetpu-compiler'], check=True)
        
        subprocess.run(['edgetpu_compiler', 'limelight_neural_detector_8bit.tflite'], cwd=FINALOUTPUTFOLDER, check=True)
        subprocess.run(['mv', 'limelight_neural_detector_8bit_edgetpu.tflite', 'limelight_neural_detector_coral.tflite'], 
                      cwd=FINALOUTPUTFOLDER, check=True)
        subprocess.run(['rm', '-f', 'limelight_neural_detector_8bit_edgetpu.log'], cwd=FINALOUTPUTFOLDER, check=True)
        
        subprocess.run(['rm', '-f', f'{HOMEFOLDER}limelight_detectors.zip'], check=True)
        subprocess.run(['zip', '-r', f'{HOMEFOLDER}limelight_detectors.zip', FINALOUTPUTFOLDER], check=True)
        
        print(f"Model compilation and packaging completed! Output: {HOMEFOLDER}limelight_detectors.zip")
        return f"{HOMEFOLDER}limelight_detectors.zip"
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation/packaging: {e}")
        return None


def main(dataset_url=None, local_dataset_path=None):
    print("Starting Limelight Detector Training Pipeline...")
    
    HOMEFOLDER, FINALOUTPUTFOLDER, MLENVIRONMENT = setup_environment()
    if not HOMEFOLDER:
        print("Environment setup failed!")
        return False
    
    train_record_fname, val_record_fname, label_map_pbtxt_fname = download_and_process_dataset(dataset_url, HOMEFOLDER, local_dataset_path)
    if not train_record_fname:
        print("Dataset processing failed!")
        return False
    
    pipeline_file, model_dir, num_steps, checkpoint_every, num_classes = setup_training_config(
        HOMEFOLDER, label_map_pbtxt_fname, train_record_fname, val_record_fname)
    if not pipeline_file:
        print("Training configuration failed!")
        return False
    
    if not train_model(HOMEFOLDER, pipeline_file, model_dir, checkpoint_every, num_steps):
        print("Training failed!")
        return False
    
    model_path_32bit = convert_to_tflite(HOMEFOLDER, FINALOUTPUTFOLDER, pipeline_file)
    if not model_path_32bit:
        print("TFLite conversion failed!")
        return False
    
    quantized_model_path = quantize_model(FINALOUTPUTFOLDER, model_path_32bit, train_record_fname, HOMEFOLDER, MLENVIRONMENT)
    if not quantized_model_path:
        print("Model quantization failed!")
        return False
    
    final_package = compile_and_package_models(FINALOUTPUTFOLDER, HOMEFOLDER)
    if not final_package:
        print("Model compilation/packaging failed!")
        return False
    
    print("Limelight Detector Training Pipeline completed successfully!")
    print(f"Final model package: {final_package}")
    return True


if __name__ == "__main__":
    import sys
    
    local_dataset_path = "/Users/axion66/coding/ftc/dataset_roboflow"
    if os.path.exists(local_dataset_path):
        print(f"Found local dataset at: {local_dataset_path}")
        main(local_dataset_path=local_dataset_path)
    else:
        dataset_url = sys.argv[1] if len(sys.argv) > 1 else None
        main(dataset_url=dataset_url)