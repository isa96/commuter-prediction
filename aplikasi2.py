import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

title = 'Predict Amount of Commuter Passenger in Indonesia 🚉'
subtitle = 'Predict Amount of Commuter Passenger in Indonesia using machine learning 🚄🚄 '

def main():
    st.set_page_config(layout="centered", page_icon='🚉', page_title='Lets Predict Amount of Commuter Passenger!')
    st.title(title)
    st.write(subtitle)
    st.write("For more information about this project, check here: [GitHub Repo](https://github.com/PrastyaSusanto/Commuter-Prediction-App/tree/main)")

    form = st.form("Data Input")
    Region = form.selectbox('Region', ['Jabodetabek', 'Non Jabodetabek (Jawa)', 'Jawa (Jabodetabek+Non Jabodetabek)', 'Sumatera'])
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Kode Wilayah': Region,
            'Tanggal Relatif': pd.date_range(start=start_date, end=end_date).to_list()
        }
        data = pd.DataFrame(data)

        data['Kode Wilayah'] = data['Kode Wilayah'].replace({'Jabodetabek': 0, 'Non Jabodetabek (Jawa)': 2, 'Jawa (Jabodetabek+Non Jabodetabek)': 1, 'Sumatera': 3})

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Relatif'] = (pd.to_datetime(data['Tanggal Relatif']) - pd.to_datetime('2006-01-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        predictions = model.predict(data)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date), 'Predicted Passenger': predictions})

        # Format the predicted passenger values as integers
        results['Predicted Passenger'] = results['Predicted Passenger'].astype(int)

        # Visualize the results using matplotlib
        plt.style.use('dark_background') 
        plt.plot(results['Date'], results['Predicted Passenger'], color='royalblue')
        plt.xlabel('Date')
        plt.ylabel('Predicted Passenger')
        plt.xticks(rotation=90)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('Predicted Amount of Commuter Passenger over Time')
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks
        
        st.pyplot(plt)

        # Format the Date column in the results DataFrame
        results['Date'] = results['Date'].dt.strftime('%d-%m-%Y')

        # Optionally, you can also show the raw data in a table
        st.dataframe(results)
    

if __name__ == '__main__':
    main()
