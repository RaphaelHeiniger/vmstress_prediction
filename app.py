import streamlit as st
import pyvista as pv
import time
import matplotlib.pyplot as plt
import numpy as np
from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd

def main():
    st.set_page_config(page_title="ML Pipeline App", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Input"
    
    option = st.sidebar.radio("Go to", ("Input", "Preprocessing", "Prediction", "Results"),
                              index=(0 if st.session_state["current_page"] == "Input" else
                                     1 if st.session_state["current_page"] == "Preprocessing" else
                                     2 if st.session_state["current_page"] == "Prediction" else 3))
    
    if option == "Input":
        input_section()
    elif option == "Preprocessing":
        preprocessing_section()
    elif option == "Prediction":
        prediction_section()
    elif option == "Results":
        results_section()

def input_section():
    st.title("Input Section")
    uploaded_file = st.file_uploader("Select the LSDYNA input file:", type=["k", "key", "dyn"])
    if uploaded_file is not None:
        st.session_state["user_data"] = uploaded_file.read().decode("utf-8")
        if st.button("Confirm Upload"):
            st.session_state["current_page"] = "Preprocessing"
            keyword_file = st.session_state["user_data"]
            deck, mesh_geometry, elements = process_kwd_to_mesh(keyword_file)
            st.session_state["deck"] = deck
            st.session_state["mesh_geometry"] = mesh_geometry
            st.session_state["elements"] = elements
            st.rerun()

def preprocessing_section():
    st.title("Preprocessing Section")
    if "user_data" in st.session_state:
        deck = st.session_state["deck"]
        deck.plot(show_edges=True)
        preprocessed_data = st.session_state["user_data"].strip().lower()
        st.session_state["preprocessed_data"] = preprocessed_data
        st.write("Preprocessed Data:", preprocessed_data)
        
        # Display PyVista visualization
        #plot ls-dyna inputdeck
        #plotter = pv.Plotter()
        #mesh = pv.Sphere()
        #plotter.add_mesh(mesh)
        #stpv.pyplot(plotter.show())
        
        if st.button("Preprocess geometry"):
            st.session_state["current_page"] = "Prediction"
            #create features from mesh
            st.rerun()
    else:
        st.warning("Please upload a file in the Input section first.")

def prediction_section():
    st.title("Prediction Section")
    if "preprocessed_data" in st.session_state:
        # Model selection
        model_choice = st.radio("Select a prediction model:", ("Model A", "Model B"))
        st.session_state["selected_model"] = model_choice
        
        if st.button("Predict"):
            st.session_state["prediction_status"] = "Preparing data for model..."
            st.session_state["prediction_step"] = 0
            st.session_state["current_page"] = "Results"
            st.rerun()
    else:
        st.warning("Please preprocess the data first.")

def results_section():
    st.title("Results Section")
    if "prediction_step" in st.session_state:
        if st.session_state["prediction_step"] == 0:
            st.write("Preparing data for model...")
            time.sleep(2)
            st.session_state["prediction_step"] = 1
            st.rerun()
        elif st.session_state["prediction_step"] == 1:
            st.write("Running prediction...")
            time.sleep(2)
            st.session_state["prediction_result"] = np.random.rand(100, 100)
            st.session_state["prediction_status"] = "Prediction Finished!"
            st.session_state["prediction_step"] = 2
            st.rerun()
        elif st.session_state["prediction_step"] == 2:
            st.success(st.session_state["prediction_status"])
            fig, ax = plt.subplots()
            ax.imshow(st.session_state["prediction_result"], cmap="viridis")
            st.pyplot(fig)
    else:
        st.warning("No prediction initiated yet.")

if __name__ == "__main__":
    main()

