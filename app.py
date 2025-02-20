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
import time

from pipe_kwd_to_mesh import process_kwd_to_mesh, plot_mesh, plot_b_l
from model_a.create_features import *
from model_a.create_dataset import create_dataset
from model_a.create_prediction import ini_model, get_prediction
from plot_prediction import plot_prediction

# Set the environment variable for offscreen rendering in PyVista
#os.environ["PYVISTA_OFF_SCREEN"] = "1"
#os.environ['DISPLAY'] = ':0'
#pv.start_xvfb()
from stpyvista.utils import start_xvfb

#this is only required for the deployed version on streamlit cloud

debug = False
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
    st.sidebar.title("Stress prediction")
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
    st.title("Input")
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
    st.title("Preprocessing")
    if "mesh_geometry" in st.session_state and "mesh_topology" in st.session_state:

        # Visualize geometry
        deck = st.session_state["deck"]
        #stpyvista(deck.plot(show_edges=True, off_screen=True), key="mesh_plot")
        mesh_geometry = st.session_state["mesh_geometry"]
        mesh_topology = st.session_state["mesh_topology"]
        mesh_plotter = plot_b_l(mesh_geometry, mesh_topology)
        #os_name = platform.system()
        #if os_name == 'Windows':
        #    mesh_plotter.show()
        #else:
        stpyvista(mesh_plotter, key="mesh_plot")  
        print(mesh_geometry.shape)
        preprocessed_data = [mesh_geometry, mesh_topology]
        st.session_state["preprocessed_data"] = preprocessed_data
        st.markdown("**Legend:**")
        st.markdown("  <span style='color:red'>**Loads**</span>", unsafe_allow_html=True)
        st.markdown("  <span style='color:blue'>**Boundaries**</span>", unsafe_allow_html=True)
        if st.button("Preprocess geometry"):
            st.session_state["current_page"] = "Prediction"
            #create features from mesh
            st.rerun()
    else:
        st.warning("Please upload a file in the Input section first.")

def prediction_section():
    st.title("Prediction")

    if "preprocessed_data" in st.session_state:
        # Model selection
        model_choice = st.radio("Select a prediction model:", ("Model A", "Model B"))
        st.session_state["selected_model"] = model_choice
            
        # Require user confirmation before proceeding
        if "model_confirmed" not in st.session_state:
            st.session_state["model_confirmed"] = False

        if st.button("Confirm Model"):
            st.session_state["model_confirmed"] = True
            st.rerun()

        if st.session_state["model_confirmed"]:
            st.write(f"Processing data for {st.session_state['selected_model']}")

        #    if st.session_state["selected_model"] == "Model A":
            st.write("Prepare features:")
                
            mesh_geometry = st.session_state["mesh_geometry"]
            mesh_topology = st.session_state["mesh_topology"]
            print(mesh_geometry.shape)
                
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
                st.write(f"... edge index: {st.session_state['edge_index'].shape}")
                st.write(f"... edge attributes: {st.session_state['edge_attr'].shape}")

            if st.button("Predict"):
                st.session_state["prediction_status"] = "Preparing data for model..."
                st.session_state["prediction_step"] = 0
                st.session_state["current_page"] = "Results"
                st.rerun()
    else:
        st.warning("Please preprocess the data first.")

def results_section():
    st.title("Results")
    
    if "prediction_step" in st.session_state:
        if st.session_state["prediction_step"] == 0:
            st.write("Preparing data for model...")

            # Ensure required data is available
            if "mesh_geometry" in st.session_state and "mesh_topology" in st.session_state:
                mesh_geometry = st.session_state["mesh_geometry"]
                mesh_topology = st.session_state["mesh_topology"]

                print(mesh_geometry.shape)
                loader = create_dataset(mesh_geometry, 
                                        mesh_topology,
                                        st.session_state["boundary"], 
                                        st.session_state["loads"], 
                                        st.session_state["edge_index"], 
                                        st.session_state["edge_attr"])
                start_time = time.time()
                st.write("Running prediction...")
                model, args = ini_model()
                pred = get_prediction(loader, model, args)
                print(pred)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Store prediction in session state
                st.session_state["predicted_stress"] = pred
                st.session_state["prediction_time"] = int(elapsed_time*1000)
                st.write(f"predicted stress tensor{pred}")

                # Move to next step
                st.session_state["prediction_step"] = 1
                st.rerun()
            else:
                st.warning("Mesh geometry or topology is missing. Please go back to the Input or Preprocessing section.")

        elif st.session_state["prediction_step"] == 1:
            st.session_state["prediction_status"] = f"Prediction Finished! Took {st.session_state["prediction_time"]} ms. {round(1000/st.session_state["prediction_time"], 2)}x speed up compared to FEM solver."
            st.session_state["prediction_step"] = 2
            st.rerun()

        elif st.session_state["prediction_step"] == 2:
            st.success(st.session_state["prediction_status"])

            # Retrieve stored values
            mesh_geometry = st.session_state["mesh_geometry"]
            mesh_topology = st.session_state["mesh_topology"]
            pred_stress = st.session_state["predicted_stress"]

            # Ensure pred_stress exists before plotting
            if pred_stress is not None:
                pred_plotter = plot_prediction(mesh_geometry, mesh_topology, pred_stress)
                stpyvista(pred_plotter, key="pred_plot") 
            else:
                st.warning("Prediction data is missing.")

    else:
        st.warning("No prediction initiated yet.")


if __name__ == "__main__":
    main()

