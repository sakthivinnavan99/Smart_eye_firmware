#!/usr/bin/env python3
"""
ONNX to RKNN converter - YOLOv8 optimized version
Supports both YOLOv5 and YOLOv8 models with automatic detection
python3 convert_yolov8.py yolov8n_test.pt rk3588 fp yolov8n_test_
2.rknn 2>&1 | grep -v "W load_onnx" | head -80
"""
import sys
import os
from pathlib import Path
from rknn.api import RKNN

DATASET_PATH = 'labels.txt'
DEFAULT_RKNN_PATH = 'Output_model.rknn'
DEFAULT_QUANT = True

def detect_yolo_version(model_path):
    """Detect YOLO model version (v5 or v8) based on filename or model inspection"""
    filename = Path(model_path).name.lower()
    
    if 'yolov8' in filename or 'v8' in filename:
        return 'v8'
    elif 'yolov5' in filename or 'v5' in filename:
        return 'v5'
    else:
        # Default to v8 if unsure
        return 'v8'

def get_yolo_config(version):
    """Get YOLO version-specific configuration"""
    configs = {
        'v8': {
            'input_size': [640, 640],
            'input_names': ['images'],
            'output_names': ['output0'],
            'mean': [0, 0, 0],
            'std': [255, 255, 255],
            'opset': 12
        },
        'v5': {
            'input_size': [640, 640],
            'input_names': ['images'],
            'output_names': ['output'],
            'mean': [0, 0, 0],
            'std': [255, 255, 255],
            'opset': 12
        }
    }
    return configs.get(version, configs['v8'])

def convert_pt_to_onnx(pt_model_path, yolo_version='v8'):
    """Convert PyTorch model to ONNX format"""
    pt_path = Path(pt_model_path)
    onnx_path = pt_path.parent / f"{pt_path.stem}.onnx"
    
    print(f"--> Converting {pt_path.name} ({yolo_version.upper()}) to ONNX...")
    
    try:
        # Try ultralytics YOLO export
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_model_path))
            print("    Using ultralytics export...")
            model.export(format='onnx', imgsz=640)
            
            # Check if export was successful
            if onnx_path.exists():
                print(f"    ✓ ONNX created: {onnx_path.name}")
                return str(onnx_path)
        except Exception as e:
            print(f"    Ultralytics export failed: {e}")
            raise
    
    except Exception as e:
        print(f"    ERROR: Could not export to ONNX: {e}")
        return None

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       model_path: path to .onnx or .pt file (YOLOv5 or YOLOv8)")
        print("       platform: rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808")
        print("       dtype: i8 (int8), u8 (uint8), or fp (float32, default)")
        print("")
        print("Examples:")
        print("  python3 {} model.onnx rk3588 fp model.rknn".format(sys.argv[0]))
        print("  python3 {} yolov8n.pt rk3588 fp yolov8n.rknn".format(sys.argv[0]))
        print("  python3 {} yolov5s.pt rk3588 i8 yolov5s_int8.rknn".format(sys.argv[0]))
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid dtype: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()
    
    # Validate model file exists
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        exit(1)

    # Detect YOLO version
    yolo_version = detect_yolo_version(model_path)
    config = get_yolo_config(yolo_version)
    
    print(f"\n{'='*60}")
    print(f"YOLO Model Converter - YOLOv8 Edition")
    print(f"Detected model: YOLO{yolo_version.upper()}")
    print(f"Target platform: {platform}")
    print(f"{'='*60}\n")

    # Convert PT to ONNX if needed
    if model_path.endswith('.pt'):
        onnx_path = convert_pt_to_onnx(model_path, yolo_version)
        if not onnx_path:
            print("ERROR: Failed to convert .pt to ONNX")
            exit(1)
        model_path = onnx_path

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Config model
    print('--> Config model')
    rknn.config(
        mean_values=[config['mean']],
        std_values=[config['std']],
        target_platform=platform
    )
    print('done')

    # Load model
    print('--> Loading model')
    try:
        if model_path.endswith('.onnx'):
            ret = rknn.load_onnx(model=model_path)
        else:
            print('ERROR: Unsupported model format. Use .onnx or .pt')
            exit(1)
    except Exception as e:
        print(f'ERROR loading model: {e}')
        print('\nNOTE: If the error mentions "mct_quantizers" or custom operators,')
        print('your model may contain MCT quantization that is incompatible with RKNN.')
        exit(1)
    
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build RKNN model
    print('--> Building model')
    ret = rknn.build(
        do_quantization=do_quant,
        dataset=DATASET_PATH if do_quant and Path(DATASET_PATH).exists() else None
    )
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    print(f'\n✓ Success! RKNN model saved to: {output_path}')
    print(f'  Model: YOLO{yolo_version.upper()}')
    print(f'  Platform: {platform}')
    print(f'  Quantization: {"Yes (int8/uint8)" if do_quant else "No (float32)"}')

    rknn.release()
