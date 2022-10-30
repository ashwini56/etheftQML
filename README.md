# EiTMOS

EiTMOS is an electricity theft monitoring system that uses Quantum Machine Learning algorithms to predict the suspicious consumers in real time.

Quantum Machine Algorithm Used for our use case:
- Quantum Support Vector Machine (QSVM)
- Quantum Kernel Training Algorithm (QKT)
- Quantum Neural Network (Customized one –Called QuClassi)

### Code Structure
<pre>
- etheftQML/
   - app.py                                                          # Backend (http://localhost:5050/)
   - templates/                                                      # Frontend        
      - index.html                                                   # Landing page
      - dash.html                                                    # Dashboard
      -  style.css    
      -  styleDash.css

    - models/                                    
      - qsvcDuplicateAfterSplit.model                                # trained QSVM-ZZ Feature Map model
      - QKT389.model                                                 # trained QSVM-Customized QKT model
      - QuclassiModel.dat                                            # trained QuClassi model
      
   - Source Code/
      - QSVM with Standard ZZFeatureMap and Customized QKT.ipynb     # Source code to train QSVM (ZZ Feature map and Customized QKT)
      - QuClassiModel.ipynb                                          # Source code to train QuClassi
      
   - data/                                                           # contains time series data, dataset used to train and test the models.
   </pre>

### Requirements
- Qiskit
- Flask
- numpy
- pandas
- Pillow
- matplotlib
- http-server

### To run the software

- Extract dataTimeSeries.csv.zip and electricity_KNNImputer.csv.zip in ~/data directory
- Run the backend server by running the command python app.py in /etheftQML-main directory
- To start the frontend, run http-server in ~/templates directory and navigate to 127.0.0.1:8000

### Model training
- Extract dataTimeSeries.csv.zip and electricity_KNNImputer.csv.zip in ~/data directory
- To train a model from scratch, run corresponding notebook provided in ~/Source Code
