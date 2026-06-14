import sys
from pathlib import Path
from rknn.api import RKNN

# Calibration dataset for int8 quantization: a text file listing one image
# path per line (representative frames from the device camera work best).
DATASET_PATH = 'dataset.txt'
DEFAULT_RKNN_PATH = '../models/pathpal/yolov8.rknn'
DEFAULT_QUANT = True

# Platforms accepted on the command line. RKNN-Toolkit2 has no 'rk3582'
# target: the RK3582 (Radxa CM5 Lite) is a binned RK3588S with the same
# NPU, so its models are built as 'rk3588' and loaded with RKNNLite.
PLATFORM_ALIASES = {
    'rk3582': 'rk3588',
    'rk3588s': 'rk3588',
}
SUPPORTED_PLATFORMS = ['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588']


def normalize_platform(platform):
    platform = platform.lower()
    if platform in PLATFORM_ALIASES:
        target = PLATFORM_ALIASES[platform]
        print(f"NOTE: '{platform}' uses the RK3588 NPU - building with target_platform='{target}'")
        return target
    if platform not in SUPPORTED_PLATFORMS:
        print('ERROR: Unsupported platform: {}'.format(platform))
        print('       choose from {} (rk3582/rk3588s map to rk3588)'.format(SUPPORTED_PLATFORMS))
        exit(1)
    return platform


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3576,rk3582,rk3588]")
        print("       dtype choose from    [i8, fp]")
        exit(1)

    model_path = sys.argv[1]
    platform = normalize_platform(sys.argv[2])

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        do_quant = (model_type == 'i8')

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    if not Path(model_path).exists():
        print('ERROR: Model file not found: {}'.format(model_path))
        exit(1)

    if do_quant and not Path(DATASET_PATH).exists():
        print('ERROR: Quantization dataset not found: {}'.format(DATASET_PATH))
        print('       Create it with one calibration image path per line,')
        print('       or pass dtype "fp" to skip quantization.')
        exit(1)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant,
                     dataset=DATASET_PATH if do_quant else None)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
