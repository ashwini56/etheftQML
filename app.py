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

N = 200 # number of datapoints to test (QSVM)

qsvcRep = QSVC.load('models/qsvcDuplicateAfterSplit.model')
testDat = np.loadtxt('data/TestData.dat')
testInd = np.loadtxt('data/XtestIndices.dat')
peruData = pd.read_csv("data/electricity_KNNImputer.csv")
XPeruTestDat = peruData.iloc[testInd, :]
predDat = qsvcRep.predict(testDat[:N])


qsvcKT = QSVC.load('models/QKT389.model')
testDatKT = np.loadtxt('data/TestDataQKT389.dat')
testIndKT = np.loadtxt('data/XtestIndicesQKT389.dat')
DataKT = pd.read_csv("data/dataTimeSeries.csv")
XTestDatKT = DataKT.iloc[testIndKT, :]
predDatKT = qsvcKT.predict(testDatKT)

quClassiModel = np.loadtxt('models/QuclassiModel.dat', delimiter=',')
param0 = quClassiModel[0]
param1 = quClassiModel[1]

Ypred = []

for i in range(len(testDatKT)):    
	qc = QuantumCircuit(7,1)
	qc.h(0)
	qc.ry(param0[0],1)
	qc.rz(param0[1],1)
	qc.ry(param0[2],2)
	qc.rz(param0[3],2)
	qc.ry(param0[4],3)
	qc.rz(param0[5],3)
	qc.ry(testDatKT[i,0],4)
	qc.rz(testDatKT[i,1],4)
	qc.ry(testDatKT[i,2],5)
	qc.rz(testDatKT[i,3],5)
	qc.ry(testDatKT[i,4],6)
	qc.rz(testDatKT[i,5],6)
	qc.cswap(0,1,4)
	qc.cswap(0,2,5)
	qc.cswap(0,3,6)
	qc.h(0)
	qc.measure(0, 0)
	backend = BasicAer.get_backend("qasm_simulator")
	job = execute(qc, backend, shots=2^10, seed_simulator=1024, seed_transpiler=1024)
	counts = job.result().get_counts(qc)
	if '1' in counts:
		p = counts['1'] / (counts['1'] + counts['0'])
		s = 1 - (2*p)
	else: s = 1
	if s<=0: s = 1e-16
	prob0 = s
	qc1 = QuantumCircuit(7,1)
	qc1.h(0)
	qc1.ry(param1[0],1)
	qc1.rz(param1[1],1)
	qc1.ry(param1[2],2)
	qc1.rz(param1[3],2)
	qc1.ry(param1[4],3)
	qc1.rz(param1[5],3)
	qc1.ry(testDatKT[i,0],4)
	qc1.rz(testDatKT[i,1],4)
	qc1.ry(testDatKT[i,2],5)
	qc1.rz(testDatKT[i,3],5)
	qc1.ry(testDatKT[i,4],6)
	qc1.rz(testDatKT[i,5],6)
	qc1.cswap(0,1,4)
	qc1.cswap(0,2,5)
	qc1.cswap(0,3,6)
	qc1.h(0)
	qc1.measure(0, 0)
	backend1 = BasicAer.get_backend("qasm_simulator")
	job1 = execute(qc1, backend1, shots=2^10, seed_simulator=1024, seed_transpiler=1024)
	counts1 = job1.result().get_counts(qc1)
	if '1' in counts1:
		p1 = counts1['1'] / (counts1['1'] + counts1['0'])
		s1 = 1 - (2*p1)
	else: s1 = 1    
	if s1<=0: s1 = 1e-16
	prob1 = s1
	p0 = prob0 / (prob0 + prob1)
	p1 = prob1 / (prob0 + prob1)
	probs = np.array([p0, p1])
	Ypred.append(np.argmax(probs))
		
predDatQuClassi = np.array(Ypred)

ind = []
ind.append(0)
print(ind)



@app.route('/')
def home():
    return render_template('dash.html')
    
	
@app.route("/model", methods=["GET", "POST"])
def model():
	m = request.form.get("model")
	
	print(m)
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
	tSeries = np.array(xPeruTest[-1].iloc[i,:][100:])
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
	t1 = int(len(Xtest[-1]))
	return Response(str(t1), mimetype='text', status=200)
	
@app.route("/totPred", methods=["GET", "POST"])
def totPred():

	i = request.json
	ind.append(i)
	Xp = XPred[-1]
	t = int(len(Xp[:i+1][Xp[:i+1]==1]))
	return Response(str(t), mimetype='text', status=200)
	
@app.route("/totTrue", methods=["GET", "POST"])
def totTrue():

	i = request.json
	ind.append(i)
	Xp = XPred[-1]
	t = int(len(Xp[:i+1][Xp[:i+1]==0]))

	return Response(str(t), mimetype='text', status=200)
	
@app.route("/pred", methods=["GET", "POST"])
def pred():

	i = request.json
	ind.append(i)
	Xp = XPred[-1]
	p = Xp[i]
	if p==0:
		t = 'Legitimate'
	else : t = 'Suspicious'
	
	return Response(t, mimetype='text', status=200)


@app.route("/table", methods=["GET", "POST"])
def getSuspiciousTablec():

	i = request.json
	Xp = XPred[-1]
	xPT = xPeruTest[-1]
	
	if Xp[i]==1:
		data = np.array([[xPT.iloc[i,0], np.round(xPT.iloc[i,2:].sum(skipna=True), 2), np.round(xPT.iloc[i,2:].mean(skipna=True), 2), np.round(np.random.random(1)*1000, 2)[0], np.random.randint(200000, 1000000)]])
		d = pd.DataFrame(data=data, columns=['cons', 'sum', 'mean', 'due', 'pin'])
		out = d.to_json(orient='records')
		return Response(out, mimetype='application/json', status=200)
	else : return Response(str(""), status=204)


if __name__ == "__main__":
	app.run(host='localhost', port=5050, debug=True)
