# EiTMOS

EiTMOS is an electricity theft monitoring system that uses Quantum Machine Learning algorithms to predict the suspicious consumers in real time.

Quantum Machine Algorithm Used for our use case:
- Quantum Support Vector Machine (QSVM)
- Quantum Kernel Training Algorithm (QKT)
- Quantum Neural Network (Customized one –Called QuClassi)

### Code Structure
<pre>
- etheftQML/
   - app.py                                      # Backend (http://localhost:5050/)
   - templates/                                  # Frontend        
      - index.html                               # Landing page
      - dash.html                                # Dashboard
      -  style.css    
      -  styleDash.css

    - models/                                    
      -  qsvcDuplicateAfterSplit.model            # QSVM
      - QKT389.model                              # QKT
      - QuclassiModel.dat                         # QuClassi
   - data/                                        # contains time series data, dataset used to train and test the models.
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
- Run the backend server by running the command python app.py in /etheftQML-main directory [^1]
- To start the frontend, run http-server in ~/templates and navigate to 127.0.0.1:8000

[^1]: Time series dataset used in QSVM (/data/electricity_KNNImputer.csv) is a heavier to add to this repo. We will upload it shortly.
