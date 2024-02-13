from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Create your views here.


def first_template(request):

    return render(request, "predictions/predict_form.html")

# Import necessary libraries
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def result(request):
    if request.method == "POST":
        # Load the dataset
        df = pd.read_csv('house_predict.csv')
        X = df[['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE', 'LONGITUDE', 'LATITUDE', 'POSTED_BY_0', 'POSTED_BY_1', 'BHK_OR_RK_0', 'BHK_OR_RK_1']]
        y = df['TARGET(PRICE_IN_LACS)']

        X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=1)

        model = LinearRegression()
        model = model.fit(X_train, y_train) 

        # Extract user input from the form
        posted_by = request.POST.get('posted_by')
        under_construction = (request.POST.get('under_construction'))
        rera = (request.POST.get('rera'))
        bhk_no = (request.POST.get('bhk_no'))
        square_ft = (request.POST.get('square_ft'))
        ready_to_move = (request.POST.get('ready_to_move'))
        resale = (request.POST.get('resale'))
        longitude = (request.POST.get('longitude'))
        latitude = (request.POST.get('latitude'))
        bhk_or_rk = request.POST.get('bhk_or_rk')  

        user_input = {
            'POSTED_BY': posted_by,
            'UNDER_CONSTRUCTION': under_construction,
            'RERA': rera,
            'BHK_NO.': bhk_no,
            'BHK_OR_RK': bhk_or_rk,  # Corrected capitalization
            'SQUARE_FT': square_ft,
            'READY_TO_MOVE': ready_to_move,
            'RESALE': resale,
            'LONGITUDE': longitude,
            'LATITUDE': latitude,
        }

        if user_input['POSTED_BY'] == 'Owner':
            user_input['POSTED_BY_0'] = 0
            user_input['POSTED_BY_1'] = 1
        elif user_input['POSTED_BY'] == 'Dealer':
            user_input['POSTED_BY_0'] = 1
            user_input['POSTED_BY_1'] = 0
        else:
            user_input['POSTED_BY_0'] = 0
            user_input['POSTED_BY_1'] = 0

        # Encode 'BHK_OR_RK' input (BHK as 0, RK as 1)
        if user_input['BHK_OR_RK'] == 'BHK':  # Corrected capitalization
            user_input['BHK_OR_RK_0'] = 0
            user_input['BHK_OR_RK_1'] = 1
        elif user_input['BHK_OR_RK'] == 'RK':  # Corrected capitalization
            user_input['BHK_OR_RK_0'] = 1
            user_input['BHK_OR_RK_1'] = 0
        else:
            user_input['BHK_OR_RK_0'] = 0
            user_input['BHK_OR_RK_1'] = 0

        # Drop the original 'POSTED_BY' and 'BHK_OR_RK' columns from the user input
        user_input.pop('POSTED_BY')
        user_input.pop('BHK_OR_RK')

        user_df = pd.DataFrame(user_input, index=[0])

        # Create a NumPy array with user input and reshape it
        user_input_data = user_df.values.reshape(1, -1)

        # Use the trained model to make a prediction
        predicted_price = abs(model.predict(user_input_data)[0])
        formatted_price = "The predicted price is Rupees {:.2f} Lakhs".format(predicted_price)

        return render(request, "predictions/predict_form.html", {"predicted_price": formatted_price})

    return render(request, "predictions/predict_form.html")





