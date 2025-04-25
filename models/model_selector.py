from models.ast import ASTModelWrapper
from models.cnn import CNN
# from models.mlp_model import MLPModel

def get_model(model_name, num_labels):
    if model_name == "cnn":
        return CNN(num_labels)
    elif model_name == "mlp":
        return MLPModel(num_labels)
    elif model_name == "ast":
        return ASTModelWrapper(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels).model

    else:
        raise ValueError("Invalid model name. Choose from 'cnn', 'mlp', or 'ast'")