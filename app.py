import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
from ansys.dyna.core.lib.deck import Deck
from ansys.dyna.core.keywords import keywords
from pathlib import Path
import pandas as pd
from stpyvista import stpyvista
import platform
import os

from pipe_kwd_to_mesh import process_kwd_to_mesh, plot_mesh
from model_a.create_features import *
from model_a.create_dataset import create_dataset
from model_a.create_prediction import ini_model, get_prediction


# Set the environment variable for offscreen rendering in PyVista
#os.environ["PYVISTA_OFF_SCREEN"] = "1"
#os.environ['DISPLAY'] = ':0'
#pv.start_xvfb()
from stpyvista.utils import start_xvfb

#this is only required for the deployed version on streamlit cloud

debug = True
os_name = platform.system()

if os_name == 'Windows':
    pass
else:
    if "IS_XVFB_RUNNING" not in st.session_state:
        start_xvfb()
        st.session_state.IS_XVFB_RUNNING = True 
    ##################################################################

def main():
    st.set_page_config(page_title="Stress prediction", layout="wide")
    
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
            deck, mesh_geometry, mesh_topology = process_kwd_to_mesh(keyword_file)
            st.session_state["deck"] = deck
            st.session_state["mesh_geometry"] = mesh_geometry
            st.session_state["mesh_topology"] = mesh_topology
            st.rerun()

def preprocessing_section():
    st.title("Preprocessing Section")
    if "mesh_geometry" in st.session_state and "mesh_topology" in st.session_state:

        # Visualize geometry
        deck = st.session_state["deck"]
        #stpyvista(deck.plot(show_edges=True, off_screen=True), key="mesh_plot")
        mesh_geometry = st.session_state["mesh_geometry"]
        mesh_topology = st.session_state["mesh_topology"]
        mesh_plotter = plot_mesh(mesh_geometry, mesh_topology)
        os_name = platform.system()
        if os_name == 'Windows':
            mesh_plotter.show()
        else:
            stpyvista(mesh_plotter, key="mesh_plot")  

        preprocessed_data = [mesh_geometry, mesh_topology]
        st.session_state["preprocessed_data"] = preprocessed_data

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
        
        if model_choice == 'Model A':
            st.write(f"Processing data for {model_choice}")
            #MODEL A
            ############################################
            st.write("Prepare features:")

            mesh_geometry = st.session_state["mesh_geometry"]
            mesh_topology = st.session_state["mesh_topology"]

            st.write(f".. apply boundary conditions")
            constrain_boxes = [((0, 0), (250, 0.0001)),
                ((249.999, 0), (250.0001, 500))]
            st.session_state["boundary"] = apply_boundary_conditions(mesh_geometry, boxes=constrain_boxes)
            if debug:
                st.dataframe(st.session_state["boundary"])
            st.write(f".. apply external loads")
            load_boxes = [((0, 499.9999), (250, 500.0001), (100, 100)),
                ((0, 0), (1, 500.0001), (-30, 0))]
            st.session_state["loads"] = apply_external_loads(mesh_geometry, boxes=load_boxes)
            if debug:
                st.dataframe(st.session_state["loads"])

            st.write(f".. create connectivity and edge features")
            st.session_state["edge_index"], st.session_state["edge_attr"] = create_edge_features(mesh_geometry, mesh_topology)
            if debug:
                st.write(f"... edge index: {st.session_state["edge_index"].shape}")
                st.write(f"... edge attributes: {st.session_state["edge_attr"].shape}")

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


            loader = create_dataset(st.session_state["mesh_geometry"], 
                                    st.session_state["mesh_topology"],
                                    st.session_state["boundary"], 
                                    st.session_state["loads"], 
                                    st.session_state["edge_index"], 
                                    st.session_state["edge_attr"])
            model, args = ini_model()
            pred = get_prediction(loader, model, args)
            st.write(f"... edge attributes: {pred}")
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
            #fig, ax = plt.subplots()
            #ax.imshow(st.session_state["prediction_result"], cmap="viridis")
            #st.pyplot(fig)
    else:
        st.warning("No prediction initiated yet.")

if __name__ == "__main__":
    main()

