from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

clf = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(25)
    # set the numerical input as they are
    enc_input[0] = data['AGE']
    #enc_input[1] = data['AGE']

    cols = ['AGE', 'MARSTAT_NEVER MARRIED','MARSTAT_NOW MARRIED', 'MARSTAT_SEPARATED', 'MARSTAT_DIVORCED OR WIDOWED',
    'EMPLOY_FULL TIME', 'EMPLOY_PART TIME', 'EMPLOY_UNEMPLOYED','EMPLOY_NOT IN LABOR FORCE','NOPRIOR_NO PRIOR TREATMENT EPISODES',
    'NOPRIOR_ONE PRIOR TREATMENT EPISODES', 'NOPRIOR_TWO PRIOR TREATMENT EPISODES','NOPRIOR_THREE PRIOR TREATMENT EPISODES',
    'NOPRIOR_FOUR PRIOR TREATMENT EPISODES','NOPRIOR_FIVE OR MORE PRIOR TREATMENT EPISODES', 'LIVARAG_HOMELESS', 'LIVARAG_DEPENDENT LIVING',
    'LIVARAG_INDEPENDENT LIVING','GENDER_MALE', 'GENDER_FEMALE','EDUC_EIGHT YEARS OR LESS','EDUC_SOME HIGH SCHOOL','EDUC_HIGH SCHOOL GRADUATE',
    'EDUC_SOME COLLEGE','EDUC_COLLEGE GRAD OR HIGHER']


    redefinded_user_input = 'MARSTAT_'+data['MARSTAT']
    # search for the index in columns name list
    MARSTAT_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[MARSTAT_column_index] = 1
    ##################### EMPLOY ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'EMPLOY_'+data['EMPLOY']
    # search for the index in columns name list
    EMPLOY_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[EMPLOY_column_index] = 1
    ##################### NOPRIOR ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'NOPRIOR_'+data['NOPRIOR']
    # search for the index in columns name list
    NOPRIOR_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[NOPRIOR_column_index] = 1
    ##################### LIVARAG ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'LIVARAG_'+data['LIVARAG']
    # search for the index in columns name list
    LIVARAG_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[LIVARAG_column_index] = 1
    ##################### GENDER ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'GENDER_'+data['GENDER']
    # search for the index in columns name list
    GENDER_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[GENDER_column_index] = 1

    ##################### EDUC ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'EDUC_'+data['EDUC']
    # search for the index in columns name list
    EDUC_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[EDUC_column_index] = 1
    return enc_input

@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    AGE = result['AGE']
    MARSTAT = result['MARSTAT']
    EMPLOY = result['EMPLOY']
    NOPRIOR = result['NOPRIOR']
    LIVARAG = result['LIVARAG']
    GENDER = result['GENDER']
    EDUC = result['EDUC']

    user_input = {'AGE': AGE, 'MARSTAT': MARSTAT, 'EMPLOY': EMPLOY, 'NOPRIOR': NOPRIOR,
    'LIVARAG': LIVARAG, 'GENDER': GENDER, 'EDUC': EDUC}
    #print(user_input)

    opioid_pred = clf.predict_proba([input_to_one_hot(user_input)])
    a=round((opioid_pred[0][1])*100)
    #return jsonify({'opioid_pred': opioid_pred[0][1]})

    return json.dumps("The probability of opioid misuse is {value} %".format(value=a));

if __name__ == '__main__':
    app.run(port=5050, debug=True)