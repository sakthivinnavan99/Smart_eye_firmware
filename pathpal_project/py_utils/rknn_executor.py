try:
    from rknn.api import RKNN
except ImportError:
    print('import rknn failed,try to import rknnlite')
    RKNNLite = True
    from rknnlite.api import RKNNLite as RKNN


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if RKNNLite or target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result