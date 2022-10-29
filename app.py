import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3

import json

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from PIL import Image
import base64
import io

from qiskit import assemble, Aer, transpile, execute
from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.visualization import *
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit.algorithms.optimizers import SPSA


import time

import scipy.stats as stats

import seaborn

import sklearn
from sklearn import metrics

import pandas as pd
import datetime as dt

from flask import Flask, render_template
from flask import send_file
import flask
from flask_cors import CORS
from flask import request
from flask import Response



app = Flask(__name__)
CORS(app)


qsvcRep = [] 
Xtest =  []
XtestInd = [] 

xPeru = [] 

xPeruTest = [] 
XPred = [] 

qsvcRep = QSVC.load('data/qsvcDuplicateAfterSplit.model')
testDat = np.loadtxt('data/TestData.dat')
testInd = np.loadtxt('data/XtestIndices.dat')
peruData = pd.read_csv("data/electricity_KNNImputer.csv")
XPeruTestDat = peruData.iloc[testInd, :]
predDat = qsvcRep.predict(testDat[:1000])


qsvcKT = QSVC.load('data/QKT389.model')
testDatKT = np.loadtxt('data/TestDataQKT389.dat')
testIndKT = np.loadtxt('data/XtestIndicesQKT389.dat')
DataKT = pd.read_csv("data/dataTimeSeries.csv")
XTestDatKT = DataKT.iloc[testIndKT, :]
predDatKT = qsvcKT.predict(testDatKT[:150])

quClassiModel = np.loadtxt('data/QuclassiModel.dat', delimiter=',')
param0 = quClassiModel[0]
param0 = quClassiModel[1]
predDatQuClassi = predQuClassi(testDatKT, param0, param1)

ind = []
ind.append(0)
print(ind)

def predQuClassi():
	Ypred = []

	for i in range(len(Xtest)):    
    	qc0 = constructCircuit(param0, Xtest[i])
    	prob0 = meas(qc0)
	    qc1 = constructCircuit(param1, Xtest[i])
    	prob1 = meas(qc1)
	    p0 = prob0 / (prob0 + prob1)
    	p1 = prob1 / (prob0 + prob1)
	    probs = np.array([p0, p1])
    	Ypred.append(np.argmax(probs))
    
    return Ypred

@app.route('/')
def home():
    return render_template('dash.html')

@app.route("/model", methods=["GET", "POST"])
def model():
	m = request.form.get("model")
	
	print(m)
	print(Xtest)
	if m=='qsvm':
		Xtest.append(testDat)
		XtestInd.append(testInd)
		xPeru.append(peruData)
		xPeruTest.append(XPeruTestDat.iloc[:,2:])
		XPred.append(predDat)

	if m=='qkt':
		Xtest.append(testDatKT)
		XtestInd.append(testIndKT)
		xPeru.append(DataKT)
		xPeruTest.append(XTestDatKT)
		XPred.append(predDatKT)

	if m=='quclassi':
		Xtest.append(testDatKT)
		XtestInd.append(testIndKT)
		xPeru.append(DataKT)
		xPeruTest.append(XTestDatKT)
		XPred.append(predDatQuClassi)
		
	return Response(str(""), status=204)

	
	
@app.route("/tseries.png", methods=["GET", "POST"])
def tseries():

	i = ind[-1]
	tSeries = np.array(xPeruTest[0].iloc[i,:][100:])
	fig = Figure(figsize =(8, 3))
	axis = fig.add_subplot(1,1,1)
	axis.plot(tSeries)
	
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(output.getvalue()).decode('utf8')

	return Response(output.getvalue(), mimetype='image/png')

@app.route("/tot", methods=["GET", "POST"])
def totCon():

	i = request.json 
	ind.append(i)
	t1 = int(len(Xtest[0]))
	return Response(str(t1), mimetype='text', status=200)
	
@app.route("/totPred", methods=["GET", "POST"])
def totPred():

	i = request.json
	ind.append(i)
	Xp = XPred[0]
	t = int(len(Xp[:i+1][Xp[:i+1]==1]))
	return Response(str(t), mimetype='text', status=200)
	
@app.route("/totTrue", methods=["GET", "POST"])
def totTrue():

	i = request.json
	ind.append(i)
	Xp = XPred[0]
	t = int(len(Xp[:i+1][Xp[:i+1]==0]))

	return Response(str(t), mimetype='text', status=200)
	
@app.route("/pred", methods=["GET", "POST"])
def pred():

	i = request.json
	ind.append(i)
	Xp = XPred[0]
	p = Xp[i]
	if p==0:
		t = 'Legitimate'
	else : t = 'Suspicious'
	
	return Response(t, mimetype='text', status=200)


@app.route("/table", methods=["GET", "POST"])
def getSuspiciousTablec():

	i = request.json
	Xp = XPred[0]
	xPT = xPeruTest[0]
	
	if Xp[i]==1:
		data = np.array([[xPT.iloc[i,0], np.round(xPT.iloc[i,2:].sum(skipna=True), 2), np.round(xPT.iloc[i,2:].mean(skipna=True), 2), np.round(np.random.random(1)*1000, 2)[0], np.random.randint(200000, 1000000)]])
		d = pd.DataFrame(data=data, columns=['cons', 'sum', 'mean', 'due', 'pin'])
		out = d.to_json(orient='records')
		return Response(out, mimetype='application/json', status=200)
	else : return Response(str(""), status=204)


if __name__ == "__main__":
	app.run(host='localhost', port=5050, debug=True)
