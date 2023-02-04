# importations
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
from pretraitement.pre_treat import getTokenizedWord
import os
import json
modelFileName = "predictors"

app = Flask(__name__)
CORS(app)
#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route("/api/", methods=['POST'])
@cross_origin(origins="*")
def all_models():
    if request.method == 'POST':
        data = request.json
        post = data["comment"]
        if post:
            res = {}
            print(post)
            tokenWord = getTokenizedWord(post)
            modelsFilesNames = ["DT.amod", "CNB.amod", "MNB.amod", "BNB.amod", "LDA.amod"]
            for modelName in modelsFilesNames:
                model = pickle.load(open(os.path.join(modelFileName, modelName), "rb"))
                if modelName == "LDA.amod": 
                    tokenWord = tokenWord.toarray()
                predicted = model.predict(tokenWord)
                res[modelName.split(".")[0]] = int(str(predicted[0]))
                print("prdicted: ", predicted)
            res = {
                "success": True,
                "data": res
            }
            return jsonify(res)
        else:
            res = {
                "success": False,
                "data": {},
                "msg": "empty word request"
            }
            return jsonify(res)    
        
    else:
        res = {
            "success": False,
            "data": {},
            "msg": "POST uniquement"
        }
        return jsonify(res)