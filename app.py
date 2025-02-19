import streamlit as st
import pyvista as pv
import streamlit_pyvista as stpv
import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.set_page_config(page_title="ML Pipeline App", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Input"
    
    option = st.sidebar.radio("Go to", ("Input", "Preprocessing", "Prediction"),
                              index=(0 if st.session_state["current_page"] == "Input" else
                                     1 if st.session_state["current_page"] == "Preprocessing" else 2))
    
    if option == "Input":
        input_section()
    elif option == "Preprocessing":
        preprocessing_section()
    elif option == "Prediction":
        prediction_section()

def input_section():
    st.title("Input Section")
    uploaded_file = st.file_uploader("Select the LSDYNA input file:", type=["k", "key", "dyn"])
    if uploaded_file is not None:
        st.session_state["user_data"] = uploaded_file.read().decode("utf-8")
        if st.button("Confirm Upload"):
            st.session_state["current_page"] = "Preprocessing"
            st.experimental_rerun()

def preprocessing_section():
    st.title("Preprocessing Section")
    if "user_data" in st.session_state:
        st.write("Raw Data:", st.session_state["user_data"])
        preprocessed_data = st.session_state["user_data"].strip().lower()
        st.session_state["preprocessed_data"] = preprocessed_data
        st.write("Preprocessed Data:", preprocessed_data)
        
        # Display PyVista visualization
        plotter = pv.Plotter()
        mesh = pv.Sphere()
        plotter.add_mesh(mesh)
        stpv.pyplot(plotter.show())
        
        if st.button("Run Prediction"):
            st.session_state["current_page"] = "Prediction"
            st.session_state["prediction_status"] = "Preparing data for model..."
            st.experimental_rerun()
    else:
        st.warning("Please upload a file in the Input section first.")

def prediction_section():
    st.title("Prediction Section")
    if "preprocessed_data" in st.session_state:
        st.write("Using preprocessed data for prediction:")
        
        if "prediction_status" in st.session_state:
            st.write(st.session_state["prediction_status"])
            time.sleep(2)
            st.session_state["prediction_status"] = "Running prediction..."
            st.experimental_rerun()
        elif "prediction_result" not in st.session_state:
            time.sleep(2)
            st.session_state["prediction_result"] = np.random.rand(100, 100)
            st.session_state["prediction_status"] = "Prediction Finished!"
            st.experimental_rerun()
        else:
            st.success(st.session_state["prediction_status"])
            fig, ax = plt.subplots()
            ax.imshow(st.session_state["prediction_result"], cmap="viridis")
            st.pyplot(fig)
    else:
        st.warning("Please preprocess the data first.")

if __name__ == "__main__":
    main()
