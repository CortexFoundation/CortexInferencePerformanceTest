import Inference
import json
if __name__ == "__main__":
    config = json.load(open("config.json",'r'))
    inferObj = Inference.Inference(config)
    inferObj.loadModelTest()
    inferObj.rmInvalidData(rm_img=False)
    inferObj.testInference(config["times"])
