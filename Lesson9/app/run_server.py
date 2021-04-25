import dill
import pandas as pd

dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
final_model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	global final_model
	with open(model_path, 'rb') as f:
		final_model = dill.load(f)

modelpath = "/models/ml_pipeline.dill"


@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])

def predict():
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		age, trtbps, chol, thalachh, oldpeak, caa, sex, cp, fbs, restecg, exng, slp, thall = '','','','','','','','','','','','',''
		k = [age, trtbps, chol, thalachh, oldpeak, caa, sex, cp, fbs, restecg, exng, slp, thall]
		request_json = flask.request.get_json()

		body = {'age': age,
				'sex': sex,
				'cp': trtbps,
				'chol': chol,
				'fbs': fbs,
				'restecg': restecg,
				'thalachh': thalachh,
				'exng': exng,
				'oldpeak': oldpeak,
				'slp': slp,
				'caa': caa,
				'thall': thall
				}

		for i in body[0]:
			if request_json[i]:
				body[0][i] = request_json[i]

		try:
			preds = final_model.predict_proba(pd.DataFrame(body))
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds[:, 1][0]
		data["success"] = True

	return flask.jsonify(data)


if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
