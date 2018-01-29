import InferenceStream
import json
if __name__ == "__main__":
    config = json.load(open("config.json",'r'))
    inferObj = InferenceStream.InferenceStream(config)
    inferObj.testInference(config["times"])
