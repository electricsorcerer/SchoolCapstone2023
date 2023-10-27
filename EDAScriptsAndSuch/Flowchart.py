import tensorflow as tf
from keras.models import load_model
from graphviz import Digraph
import os
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + r'C:\Users\\61416\Downloads\windows_10_msbuild_Release_graphviz-9.0.0-win32\Graphviz\\bin'

def visualize_model(model):
    dot = Digraph()

    for layer in model.layers:
        layer_type = type(layer).__name__

        # Getting layer details
        layer_config = layer.get_config()
        layer_name = layer.name
        input_shape = layer.input_shape
        output_shape = layer.output_shape

        # Add node to the graph
        dot.node(layer_name, f"{layer_name}\nType: {layer_type}\nInput: {input_shape}\nOutput: {output_shape}")

        if hasattr(layer, 'input'):
            input_layer_name = layer.input.name.split('/')[0]
            dot.edge(input_layer_name, layer_name)

    dot.render('model_flowchart', format='png', view=True)

# Load your model
model_path = 'D:\Coding\School\img_rec_proj\TheModel\military-aircraft-classifier.h5'  # Replace with your .h5 model path
loaded_model = load_model(model_path)

# Save the visualization as an image
plot_model(loaded_model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)
# Visualize the model
visualize_model(loaded_model)