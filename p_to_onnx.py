import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_to_onnx(pickle_file, onnx_file):
    # Step 1: Load the pickled model dictionary
    with open(pickle_file, 'rb') as f:
        model_dict = pickle.load(f)

    # Extract the model from the dictionary
    model = model_dict['model']

    # Step 2: Convert the model to ONNX format
    # Adjust the input shape based on your model's requirements
    input_shape = (1,  # batch size
                   128)  # replace with the actual input size of your model

    initial_type = [('float_input', FloatTensorType(input_shape))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to a file
    with open(onnx_file, 'wb') as f:
        f.write(onnx_model.SerializeToString())

# Example usage
convert_to_onnx('model.p', 'model.onnx')
