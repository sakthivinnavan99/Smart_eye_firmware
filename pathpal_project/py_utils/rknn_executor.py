from rknnlite.api import RKNNLite


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNNLite()

        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    def run(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return self.rknn.inference(inputs=inputs)
