import Inference
import json
if __name__ == "__main__":
    config = json.load(open("config.json",'r'))
    inferObj = Inference(config)
    inferObj.loadModel()
    inferObj.rmInvalidData()
    inferObj.testInference()
